"""Cellular Potts model of a confluent tissue.

The tissue lies on a square, triangular, or disordered lattice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, SR, oo
from sage.sets.real_set import RealSet
from scipy.spatial import Voronoi

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import SamplingInteraction
from ....core.components.spaces import (
    BaseNeighborhood,
    BaseSpace,
    DiscreteTime,
    LatticeSpace,
    MooreNeighborhood,
    NetworkSpace,
    TriangularNeighborhood,
)
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RegionScoreSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array, Indices

MATRIX_AXES = 2

LATTICES = ('square', 'triangular', 'disordered')

DEFAULT_ADHESION = ((1.0, 2.0), (2.0, 1.5))

TILES = 3


class TissueSiteElement(Element):
    """Site of a lattice tissue, holding the label of the cell that covers it.

    The sites of one label form a cell of the cellular Potts model, and `CellAdhesionInteraction` moves the boundaries
    between cells by copying the label of a neighbouring site.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        cell:
            The initial label of the cell that covers the site, from 0.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.biology.physiology import TissueSiteElement
        >>> TissueSiteElement('site', cell=3).cell
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cell: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cell = cell


class CellAdhesionInteraction(SamplingInteraction):
    """Cellular Potts sampling of a tissue, whose cells move their boundaries.

    The cells move under adhesion and area constraints.

    The energy of the Graner-Glazier model is `sum_<i,j> w_ij J(t(s_i), t(s_j)) [s_i != s_j] + lambda sum_s (a_s -
    A)^2 + mu sum_s (p_s - P)^2`, where `s_i` is the cell of site `i`, `t(s)` the kind of cell `s`, `J` the adhesion
    energy per unit contact between kinds, `w_ij` the weight of the contact of two sites, `a_s` the area of a cell, and
    `p_s` its perimeter, the weighted number of its contacts with other cells. Every sweep proposes `hits` copies per
    site of the cell of a random neighbour into a random site, accepted by the Metropolis rule at the temperature. A
    heterotypic adhesion above the mean of the homotypic ones sorts the kinds, the differential adhesion of Steinberg.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of sites.

        kinds:
            The kind of every cell, by label.

        adhesion:
            The adhesion energy per unit contact between cells of every pair of kinds, a symmetric matrix.

        temperature:
            The temperature of the copies, the amplitude of the fluctuations of the cell boundaries.

        area_strength:
            The strength `lambda` of the area constraint.

        target_area:
            The target area `A` of every cell.

        perimeter_strength:
            The strength `mu` of the perimeter constraint.

        target_perimeter:
            The target perimeter `P` of every cell.

        state:
            The name of the state that holds the cell of a site.

        rule:
            The acceptance, `'metropolis'` or `'glauber'`.

        neighborhood:
            The neighbours of a site, or `None` for the default neighbourhood of the space.

        hits:
            The number of copies proposed per site and sweep.

    Raises:
        ValidationError:
            The adhesion is not a symmetric square matrix, or a kind is not one of its rows.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CellAdhesionInteraction
        >>> adhesion = CellAdhesionInteraction('adhesion', target='tissue', kinds=(0, 1), adhesion=((1, 2), (2, 1)))
        >>> adhesion.objective.contact(np.array([0, 0]), np.array([0, 1])).tolist()
        [1.0, 2.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        kinds: Sequence[int],
        adhesion: npt.ArrayLike,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        area_strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        target_area: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        perimeter_strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        target_perimeter: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        state: str = 'cell',
        rule: str = 'metropolis',
        neighborhood: BaseNeighborhood | None = None,
        hits: int = 1,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=RegionScoreSpec(self._get_contact, self._get_area, self._get_perimeter, hits),
            move='copy',
            rule=rule,
            neighborhood=neighborhood,
        )
        self.kinds = kinds
        self.adhesion = adhesion
        self.temperature = temperature
        self.area_strength = area_strength
        self.target_area = target_area
        self.perimeter_strength = perimeter_strength
        self.target_perimeter = target_perimeter
        self.hits = hits

    def _get_intensity(self, namespace: SimpleNamespace) -> object:
        """Return the intensity of the acceptance, one over the temperature."""
        return 1 / SR(self._get_parameter_symbol('temperature', namespace))

    def _get_contact(self, first: Indices, second: Indices) -> Array:
        """Return the adhesion energy per unit contact between the cells of two arrays.

        The arrays hold labels of cells.
        """
        kinds = np.asarray(self.kinds, dtype=np.intp)
        return np.asarray(self.adhesion, dtype=float)[kinds[first], kinds[second]]

    def _get_area(self, labels: Indices, area_strength: float, target_area: float) -> tuple[float, float]:
        """Return the strength and the target of the area constraint.

        They are the same for every cell.
        """
        del labels
        return area_strength, target_area

    def _get_perimeter(
        self,
        labels: Indices,
        perimeter_strength: float,
        target_perimeter: float,
    ) -> tuple[float, float]:
        """Return the strength and the target of the perimeter constraint.

        They are the same for every cell.
        """
        del labels
        return perimeter_strength, target_perimeter

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an invalid adhesion or kind.

        The adhesion must be a symmetric square matrix, and every kind one of its rows.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The adhesion is not a symmetric square matrix, or a kind is not one of its rows.
        """
        super()._prepare(context)
        adhesion = np.asarray(self.adhesion, dtype=float)
        if (
            adhesion.ndim != MATRIX_AXES
            or adhesion.shape[0] != adhesion.shape[1]
            or not np.allclose(adhesion, adhesion.T)
        ):
            error_msg = f'The adhesion of {self.label!r} is a symmetric matrix over the kinds, but got {adhesion!r}.'
            raise ValidationError(error_msg)
        kinds = np.asarray(self.kinds)
        if kinds.ndim != 1 or (kinds.size and (kinds.min() < 0 or kinds.max() >= adhesion.shape[0])):
            error_msg = f'The kinds of {self.label!r} are rows of the adhesion, from 0 to {adhesion.shape[0] - 1}.'
            raise ValidationError(error_msg)


def _contact_pairs(space: LatticeSpace | NetworkSpace) -> tuple[Indices, Indices, Array]:
    """Return the two sites and the weight of every contact of the tissue.

    Each contact appears in both directions.
    """
    space.prepare()
    if isinstance(space, NetworkSpace):
        first, second = space.edges_[None]
        weights = np.asarray([weight for *_, weight in space.graph.edges(data='weight', default=1.0)], dtype=float)
        return np.concatenate([first, second]), np.concatenate([second, first]), np.concatenate([weights, weights])
    neighborhood = TriangularNeighborhood() if space.geometry == 'triangular' else MooreNeighborhood()
    coordinates = np.indices(space.shape_).reshape(2, -1)
    sites = np.arange(space.size_)
    pairs = [
        (sites, np.ravel_multi_index(tuple(coordinates + np.asarray(offset)[:, None]), space.shape_, mode='wrap'))
        for offset in neighborhood.prepare().offsets_
    ]
    first, second = (np.concatenate(items) for items in zip(*pairs, strict=True))
    return first, second, np.ones(first.size)


def _homotypic_contacts(
    pairs: tuple[Indices, Indices, Array],
    kinds: Indices,
    t: float,
    values: Mapping[str, object],
) -> float:
    """Return the share of the contact between different cells of the same kind.

    It is the share of the contact between different cells that joins cells of the same kind.
    """
    del t
    first, second, weights = pairs
    cells = np.rint(np.ravel(np.asarray(values['tissue.cell'], dtype=float))).astype(np.intp)
    boundary = cells[first] != cells[second]
    same = kinds[cells[first]] == kinds[cells[second]]
    return float((weights * boundary * same).sum() / max((weights * boundary).sum(), np.finfo(float).tiny))


def _area_deviation(sizes: Array, target_area: float, t: float, values: Mapping[str, object]) -> float:
    """Return the mean absolute deviation of the cell areas from the target area.

    It is relative to the target area.
    """
    del t
    cells = np.rint(np.ravel(np.asarray(values['tissue.cell'], dtype=float))).astype(np.intp)
    areas = np.bincount(cells, weights=sizes)
    return float(np.abs(areas - target_area).mean() / target_area)


def _cellular_potts_observables(system: System) -> list[ObservableSpec]:
    """Return the share of homotypic contacts and the deviation of the areas.

    The deviation is that of the areas from their target.
    """
    tissue = next(element for element in system.elements if element.label == 'tissue')
    adhesion = next(part for part in system.interactions if isinstance(part, CellAdhesionInteraction))
    space = tissue.space.prepare()
    sizes = np.ones(space.size_)
    if isinstance(space, NetworkSpace):
        sizes = np.asarray([volume for _, volume in space.graph.nodes(data='volume', default=1.0)], dtype=float)
    kinds = np.asarray(adhesion.kinds, dtype=np.intp)
    return [
        ObservableSpec('homotypic_contacts', function=partial(_homotypic_contacts, _contact_pairs(space), kinds)),
        ObservableSpec('area_deviation', function=partial(_area_deviation, sizes, float(adhesion.target_area))),
    ]


class CellularPottsSystem(BaseSystem):
    """The cellular Potts model of a confluent tissue sorting by differential adhesion.

    Its cells are of several kinds.

    Every site of a periodic lattice belongs to one cell, and the cells start as rectangles of `cell_shape` sites, whose
    area is the target area, with kinds in equal shares at random. Label copies sample the energy `sum_<i,j> w_ij J(t_i,
    t_j) [s_i != s_j] + lambda sum_s (a_s - A)^2 + mu sum_s (p_s - P)^2` of Nemati and de Graaf (2024), Eq. 3, extended
    to the adhesion `J` between kinds of Graner and Glazier (1992). The square lattice reads the Moore neighbourhood and
    the triangular lattice the six nearest neighbours, both with unit weights, and the disordered lattice is the
    periodic Voronoi tessellation of one jittered point per unit area, whose sites have the area of their Voronoi cell
    and touch along sides of weight `w_ij`, the length of the side over the mean length. A heterotypic adhesion above
    the mean of the homotypic ones raises the share of contacts between cells of one kind. The system has the states
    `tissue.cell`.

    Args:
        shape:
            The number of rows and columns of the lattice, or of points of the disordered lattice, multiples of the
            cell shape.

        lattice:
            `'square'`, `'triangular'`, or `'disordered'`.

        cell_shape:
            The rows and columns of the initial rectangle of every cell, whose area is the target area.

        adhesion:
            The adhesion energy per unit contact between kinds, a symmetric matrix with one row per kind.

        temperature:
            The temperature of the label copies.

        area_strength:
            The strength `lambda` of the area constraint.

        perimeter_strength:
            The strength `mu` of the perimeter constraint.

        target_perimeter:
            The target perimeter `P`, the weighted number of contacts of a cell with other cells.

        hits:
            The number of copies proposed per site and sweep.

        random_state:
            The seed or generator of the kinds of the cells and of the disordered lattice.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The lattice is unknown, or the shape is not a multiple of the cell shape.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CellularPottsSystem
        >>> system = CellularPottsSystem((10, 16)).prepare().system_
        >>> system.interactions[0].kinds
        (0, 0, 1, 1)
        >>> system = CellularPottsSystem((10, 16), lattice='disordered').prepare().system_
        >>> system.elements[0].space.graph.number_of_nodes()
        160
    """

    field = 'biology/physiology'
    topics = ('cellular potts', 'tissue', 'cell sorting', 'differential adhesion', 'lattice', 'monte carlo')
    references = ('Nemati and de Graaf 2024, arXiv 2404.09055', 'Graner and Glazier 1992')
    until = 2000
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (40, 40),
        *,
        lattice: str = 'square',
        cell_shape: Sequence[int] = (5, 8),
        adhesion: npt.ArrayLike = DEFAULT_ADHESION,
        temperature: float = 2.0,
        area_strength: float = 1.0,
        perimeter_strength: float = 0.0,
        target_perimeter: float = 0.0,
        hits: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.lattice = lattice
        self.cell_shape = cell_shape
        self.adhesion = adhesion
        self.temperature = temperature
        self.area_strength = area_strength
        self.perimeter_strength = perimeter_strength
        self.target_perimeter = target_perimeter
        self.hits = hits
        self.random_state = random_state

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the observables of the built parts.

        The observables are the homotypic contacts and the deviation of the areas.

        Returns:
            settings:
                The tuned run settings by name.
        """
        settings = super().get_run_settings()
        settings['observables'] = _cellular_potts_observables(self._build_system())
        return settings

    @staticmethod
    def _disordered_lattice(shape: tuple[int, int], rng: np.random.Generator) -> nx.Graph:
        """Return the periodic Voronoi tessellation of jittered points as a graph of sites.

        There is one point per unit area.

        Every node has the position `'pos'` of its point and the area `'volume'` of its Voronoi cell, and every edge
        joins two cells that share a side, with the length of the side over the mean length as its `'weight'`.
        """
        rows, columns = shape
        grid = np.stack(np.meshgrid(np.arange(columns), np.arange(rows)), axis=-1).reshape(-1, 2).astype(float)
        points = grid + rng.random(grid.shape)
        shifts = np.array([(x, y) for y in (0, -1, 1) for x in (0, -1, 1)], dtype=float) * (columns, rows)
        diagram = Voronoi((points[None] + shifts[:, None]).reshape(-1, 2))
        count = len(points)
        graph = nx.Graph()
        for node, point in enumerate(points):
            corners = diagram.vertices[diagram.regions[diagram.point_region[node]]]
            order = np.argsort(np.arctan2(*(corners - point).T[::-1]))
            x, y = corners[order].T
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            graph.add_node(node, pos=tuple(point), volume=area)
        for (first, second), corners in zip(diagram.ridge_points, diagram.ridge_vertices, strict=True):
            if min(first, second) < count:
                length = float(np.linalg.norm(np.subtract(*diagram.vertices[corners])))
                graph.add_edge(int(first % count), int(second % count), length=length)
        mean = np.mean([length for *_, length in graph.edges(data='length')])
        for *_, data in graph.edges(data=True):
            data['weight'] = data.pop('length') / mean
        return graph

    def _build_system(self) -> System:
        """Return the parts of the cellular Potts model of a confluent tissue."""
        rows, columns = (int(size) for size in self.shape)
        height, width = (int(size) for size in self.cell_shape)
        if self.lattice not in LATTICES:
            error_msg = (
                f'The lattice of the cellular Potts model is one of {", ".join(LATTICES)}, but got {self.lattice!r}.'
            )
            raise ValidationError(error_msg)
        if rows % height or columns % width:
            error_msg = (
                f'The shape {(rows, columns)} of the tissue is not a multiple of the cell shape {(height, width)}.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        count = (rows // height) * (columns // width)
        kinds = np.asarray(self.adhesion, dtype=float).shape[0]
        kind_of = tuple(int(kind) for kind in rng.permutation(np.arange(count) % kinds))
        if self.lattice == 'disordered':
            graph = self._disordered_lattice((rows, columns), rng)
            x, y = np.asarray([point for _, point in graph.nodes(data='pos')]).T
            space: LatticeSpace | NetworkSpace = NetworkSpace(graph)
        else:
            y, x = np.indices((rows, columns)).reshape(2, -1).astype(float)
            space = LatticeSpace((rows, columns), geometry=self.lattice)
        cells = (y // height).astype(int) * (columns // width) + (x // width).astype(int)
        tissue = LatticeElement('tissue', element=TissueSiteElement, space=space, values={'cell': cells.astype(float)})
        sampling = CellAdhesionInteraction(
            'adhesion',
            target='tissue',
            kinds=kind_of,
            adhesion=self.adhesion,
            temperature=self.temperature,
            area_strength=self.area_strength,
            target_area=float(height * width),
            perimeter_strength=self.perimeter_strength,
            target_perimeter=self.target_perimeter,
            hits=self.hits,
        )
        return System(self.label, elements=[tissue], interactions=[sampling], time=DiscreteTime())
