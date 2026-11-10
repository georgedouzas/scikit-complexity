"""Cellular Potts model of a confluent tissue.

The tissue lies on a square, triangular, or disordered lattice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numba
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, SR, oo
from sage.sets.real_set import RealSet
from scipy.spatial import Voronoi

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import SamplingInteraction
from ....core.components.laws import ArrayUpdateLaw, BaseLaw
from ....core.components.spaces import (
    AdjacencyNeighborhood,
    BallNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
)
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array, Indices

MATRIX_AXES = 2
LATTICES = ('square', 'triangular', 'disordered')
DEFAULT_ADHESION = ((1.0, 2.0), (2.0, 1.5))
TILES = 3


@numba.njit(cache=True)
def _elastic_change(strength: float, target: float, total: float, step: float) -> float:  # pragma: no cover
    """Return the change of `strength * (total - target)^2` when the total grows.

    The total grows by a step.
    """
    return strength * step * (2.0 * (total - target) + step)


@numba.njit(cache=True)
def _copy_labels(
    labels: Indices,
    contacts: tuple[Indices, Indices, Array, Array],
    scores: tuple[Array, Array, Array],
    proposals: tuple[Indices, Array, Array],
    acceptance: tuple[float, bool],
) -> None:  # pragma: no cover
    """Propose label copies at the given sites and accept them in place.

    It keeps the totals of every label. Every attempt copies the label of a neighbour of a site, the neighbour at the
    fraction `picks` of its neighbours, into the site. The contacts are the neighbours of every site, their number, the
    weights of the contacts, and the volumes of the sites. The change of the objective sums the contact energies
    `contact` between labels weighted by the contacts of the site, and the change of the elastic constraints of the two
    labels, of strengths `constraints[0]` and targets `constraints[1]` on the volumes `totals[0]`, and of strengths
    `constraints[2]` and targets `constraints[3]` on the surfaces `totals[1]`. An accepted copy updates the totals of
    the two labels only. The proposals are the sites, the picks, and the uniform numbers of the acceptance, and the
    acceptance the intensity and whether the rule is the Metropolis rule.
    """
    neighbors, counts, weights, sizes = contacts
    contact, constraints, totals = scores
    sites, picks, draws = proposals
    intensity, metropolis = acceptance
    for attempt in range(sites.size):
        site = sites[attempt]
        count = counts[site]
        if count == 0:
            continue
        old = labels[site]
        new = labels[neighbors[site, min(int(picks[attempt] * count), count - 1)]]
        if old == new:
            continue
        change = 0.0
        old_surface = 0.0
        new_surface = 0.0
        for slot in range(count):
            other = labels[neighbors[site, slot]]
            weight = weights[site, slot]
            if other == new:
                new_surface -= weight
            else:
                change += weight * contact[new, other]
                new_surface += weight
            if other == old:
                old_surface += weight
            else:
                change -= weight * contact[old, other]
                old_surface -= weight
        size = sizes[site]
        change += _elastic_change(constraints[0, old], constraints[1, old], totals[0, old], -size)
        change += _elastic_change(constraints[0, new], constraints[1, new], totals[0, new], size)
        change += _elastic_change(constraints[2, old], constraints[3, old], totals[1, old], old_surface)
        change += _elastic_change(constraints[2, new], constraints[3, new], totals[1, new], new_surface)
        exponent = 0.0 if change == 0.0 else change * intensity
        if metropolis:
            accepted = exponent <= 0.0 or draws[attempt] < math.exp(-exponent)
        else:
            accepted = draws[attempt] < 1.0 / (1.0 + math.exp(min(exponent, 700.0)))
        if accepted:
            labels[site] = new
            totals[0, old] -= size
            totals[0, new] += size
            totals[1, old] += old_surface
            totals[1, new] += new_surface


@dataclass(frozen=True)
class RegionScoreSpec(BaseSpec):
    """Spec of an objective of a state of integer labels.

    The sites of one label form a region. The objective is `sum_<i,j> w_ij c(l_i, l_j) [l_i != l_j] + sum_l a_l (V_l -
    A_l)^2 + b_l (S_l - B_l)^2`, a cost of the boundaries between labelled regions and a quadratic penalty on the
    volume and the surface of every region. The first sum runs over the pairs of neighbouring sites `i` and `j` with
    the weight `w_ij` of their contact, `l_i` is the label of site `i`, `V_l` the volume of region `l`, the sum of the
    volumes of its sites, and `S_l` its surface, the sum of the weights of the contacts of its sites with sites of other
    labels. A label copy changes the volumes and the surfaces of only the two regions it moves a site between, so a
    sampler keeps them as totals per label that every accepted move updates.

    Args:
        contact:
            The function `contact(first, second, **parameters)` of two arrays of labels, which broadcast, and the
            parameters of the system that it names, which returns the cost per unit of boundary between regions of the
            labels, symmetric in the labels.

        volume:
            The function `volume(labels, **parameters)` of an array of labels, which returns the strength `a_l` and the
            target `A_l` of the volume constraint of every label, or `None` for no volume constraint.

        surface:
            The function `surface(labels, **parameters)` of an array of labels, which returns the strength `b_l` and
            the target `B_l` of the surface constraint of every label, or `None` for no surface constraint.

        hits:
            The number of copies proposed per site and sweep.

    Raises:
        ValidationError:
            The contact or a given constraint is not callable, or the number of hits is not positive.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.physiology import RegionScoreSpec
        >>> contact = lambda first, second: np.ones(np.broadcast(first, second).shape)
        >>> area = lambda labels: (np.ones(labels.shape), np.full(labels.shape, 16.0))
        >>> RegionScoreSpec(contact, volume=area, hits=2).hits
        2
    """

    contact: Callable[..., object]
    volume: Callable[..., object] | None = None
    surface: Callable[..., object] | None = None
    hits: int = 1

    def _check(self) -> None:
        """Raise an error for a contact or a constraint not callable, or a bad hit count.

        The number of hits must be at least 1.

        Raises:
            ValidationError:
                The contact or a given constraint is not callable, or the number of hits is not positive.
        """
        for name in ('contact', 'volume', 'surface'):
            function = getattr(self, name)
            if not callable(function) and (name == 'contact' or function is not None):
                error_msg = f'The {name} of a region score must be callable, but it is {function!r}.'
                raise ValidationError(error_msg)
        if not isinstance(self.hits, int) or self.hits < 1:
            error_msg = f'A region score proposes at least one copy per site, but it has hits={self.hits!r}.'
            raise ValidationError(error_msg)


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


class RegionSamplingInteraction(SamplingInteraction):
    """Monte Carlo sampling of regions of integer labels by copies of the labels of neighbours.

    The objective is a `RegionScoreSpec`, and the state holds integer labels from 0 on the sites of a `LatticeSpace` or
    the nodes of a `NetworkSpace`, whose sites of one label form a region. Every sweep proposes `hits` copies per site,
    each at a random site, of the label of one of its neighbours drawn at random, and accepts it from the change of the
    contact energy between regions and of the volume and surface constraints of the two regions it changes. The sampler
    keeps the volume and the surface of every region and updates them on every accepted copy. The neighbourhood is that
    of the copies and the contacts, by default the Moore neighbourhood of a square lattice, the six neighbours of a
    triangular lattice, or the edges of a network, whose edge attribute `'weight'` weighs the contact and node
    attribute `'volume'` the volume of a site. The schedule does not apply. The arguments are those of
    `SamplingInteraction`, with the move `'copy'`.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.systems.biology.physiology import RegionSamplingInteraction, RegionScoreSpec
        >>> class SiteElement(Element):
        ...     def __init__(self, label, region: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.region = region
        >>> contact = lambda first, second: np.ones(np.broadcast(first, second).shape)
        >>> area = lambda labels: (1.0, 16.0)
        >>> copies = RegionSamplingInteraction(
        ...     'copies', target='sites', state='region', objective=RegionScoreSpec(contact, area), move='copy'
        ... )
        >>> regions = np.kron(np.arange(4).reshape(2, 2), np.ones((4, 4)))
        >>> space = LatticeSpace((8, 8))
        >>> sites = GroupElement('sites', element=SiteElement, space=space, on='sites', values={'region': regions})
        >>> system = CompositeSystem(elements=[sites], interactions=[copies], time=DiscreteTime())
        >>> final = simulate(system, until=20, random_state=0).states[-1]
        >>> sorted(np.unique(final).tolist())
        [0.0, 1.0, 2.0, 3.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ArrayUpdateLaw,)

    def _check_move(self) -> None:
        """Raise an error for a move other than `'copy'`, which copies the label of a neighbour."""
        if self.move != 'copy':
            error_msg = f"The region sampling {self.label!r} moves labels by the move 'copy', not {self.move!r}."
            raise ValidationError(error_msg)

    def _check_objective(self, objective: object) -> None:
        """Raise an error for an objective that is not a region score, or a region score it cannot run."""
        if not isinstance(objective, RegionScoreSpec):
            error_msg = f'The objective of region sampling {self.label!r} must be a RegionScoreSpec.'
            raise ValidationError(error_msg)
        self._check_region_score(objective)

    def _check_region_score(self, score: RegionScoreSpec) -> None:
        """Raise an error for a region score with a bad move or the heat-bath rule.

        A bad move is a move other than a label copy.
        """
        score._check()
        if self.move != 'copy':
            error_msg = f"The region score of sampling {self.label!r} moves labels by the move 'copy'."
            raise ValidationError(error_msg)
        if self.rule == 'heat-bath':
            error_msg = f"The region score of sampling {self.label!r} takes the rule 'metropolis' or 'glauber'."
            raise ValidationError(error_msg)

    @staticmethod
    @functools.cache
    def _get_contacts(space: BaseSpace, neighborhood: BaseNeighborhood | None) -> tuple[Indices, Indices, Array, Array]:
        """Return the tables of the neighbours of every site, their contacts, and volumes.

        The tables hold the weights of the contacts and the site volumes.

        A lattice site has the neighbours of the neighbourhood, or of the default neighbourhood of the lattice, with
        contacts of weight 1 and a volume of 1. A node of a network has its neighbours along the edges of the layer of
        the neighbourhood, with the edge attribute `'weight'` of the contact and the node attribute `'volume'`, both 1
        by default.

        Args:
            space:
                The prepared lattice or network.

            neighborhood:
                The neighbourhood of the contacts, or `None` for the default neighbourhood of the space.

        Returns:
            neighbors:
                The index of every neighbour of every site, the other sites first, padded with the site itself.

            counts:
                The number of neighbours of every site.

            weights:
                The weight of the contact of every site with every neighbour, 0 for padding.

            sizes:
                The volume of every site.
        """
        over = space._get_default_neighborhood() if neighborhood is None else neighborhood
        size = space.size_
        sizes = np.ones(size)
        if isinstance(space, NetworkSpace):
            layer = over.layer if isinstance(over, AdjacencyNeighborhood) else None
            graph = space.graphs_[layer]
            indptr, others, edges = space.adjacency_[layer]
            rows = np.repeat(np.arange(size), np.diff(indptr))
            edge_weights = np.asarray([data.get('weight', 1.0) for *_, data in graph.edges(data=True)], dtype=float)
            nodes = space.graph.nodes(data='volume', default=1.0)
            sizes = np.asarray([volume for _, volume in nodes], dtype=float)
            table = np.full((size, int(np.diff(indptr).max(initial=0))), size, dtype=np.intp)
            slots = np.arange(others.size) - indptr[rows]
            table[rows, slots] = others
            values = np.zeros(table.shape)
            values[rows, slots] = edge_weights[edges] if edges.size else 0.0
        else:
            table = space._get_index_table(over)
            values = np.ones(table.shape)
        sites = np.arange(size)[:, None]
        valid = (table < size) & (table != sites)
        order = np.argsort(~valid, axis=1, kind='stable')
        valid = np.take_along_axis(valid, order, axis=1)
        neighbors = np.where(valid, np.take_along_axis(table, order, axis=1), sites)
        weights = np.where(valid, np.take_along_axis(values, order, axis=1), 0.0)
        return neighbors, valid.sum(axis=1).astype(np.intp), weights, sizes

    def _get_constraint(
        self,
        function: Callable[..., object] | None,
        labels: Indices,
        parameters: Mapping[str, object],
    ) -> tuple[Array, Array]:
        """Return the strength and the target of a constraint of every label.

        They are zero for no constraint.
        """
        if function is None:
            return np.zeros(labels.size), np.zeros(labels.size)
        constraint = np.asarray(function(labels, **self._get_keywords(function, parameters)), dtype=object)
        strengths = np.asarray(np.broadcast_to(np.asarray(constraint[0], dtype=np.float64), labels.shape), dtype=float)
        targets = np.asarray(np.broadcast_to(np.asarray(constraint[1], dtype=np.float64), labels.shape), dtype=float)
        return strengths, targets

    def _sample_regions(
        self,
        values: Array,
        *,
        random: np.random.Generator,
        time: float,
        space: BaseSpace,
        **parameters: object,
    ) -> Array:
        """Return a state of labels after one sweep of label copies.

        The sweep makes as many copies as `hits` times the sites.
        """
        rng = random
        score = getattr(self, self._get_parameter('objective'))
        if not isinstance(space, (LatticeSpace, NetworkSpace)) or values.size != space.size_:
            error_msg = f'The region score of sampling {self.label!r} reads a label per site of a lattice or a network.'
            raise ValidationError(error_msg)
        labels = np.rint(values).astype(np.intp).ravel()
        if labels.size and labels.min() < 0:
            error_msg = f'The region score of sampling {self.label!r} reads labels from 0, but got {labels.min()}.'
            raise ValidationError(error_msg)
        intensity = float(SR(self._get_intensity(SimpleNamespace(**parameters))))
        if self.cooling is not None:
            intensity /= float(SR(self.cooling(time)))
        neighbors, counts, weights, sizes = self._get_contacts(space, self.neighborhood)
        ids = np.arange(int(labels.max(initial=-1)) + 1)
        contact = score.contact(ids[:, None], ids[None, :], **self._get_keywords(score.contact, parameters))
        contact = np.ascontiguousarray(np.broadcast_to(np.asarray(contact, dtype=float), (ids.size, ids.size)))
        constraints = np.stack(
            [
                *self._get_constraint(score.volume, ids, parameters),
                *self._get_constraint(score.surface, ids, parameters),
            ],
        )
        foreign = (weights * (labels[neighbors] != labels[:, None])).sum(axis=1)
        totals = np.stack(
            [
                np.bincount(labels, weights=sizes, minlength=ids.size),
                np.bincount(labels, weights=foreign, minlength=ids.size),
            ],
        )
        attempts = score.hits * labels.size
        _copy_labels(
            labels,
            (neighbors, counts, weights, sizes),
            (contact, constraints, totals),
            (rng.integers(labels.size, size=attempts), rng.random(attempts), rng.random(attempts)),
            (intensity, self.rule == 'metropolis'),
        )
        return labels.reshape(values.shape).astype(values.dtype)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the law that samples the state by the score.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The law of the sampling.
        """
        target = getattr(self._get_member(namespace, 'target'), self.state)
        return [ArrayUpdateLaw(target, self._sample_regions, label=self.label)]


class CellAdhesionInteraction(RegionSamplingInteraction):
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

    @staticmethod
    def _get_contact_pairs(space: LatticeSpace | NetworkSpace) -> tuple[Indices, Indices, Array]:
        """Return the two sites and the weight of every contact of the tissue.

        Each contact appears in both directions.
        """
        space.prepare()
        if isinstance(space, NetworkSpace):
            first, second = space.edges_[None]
            weights = np.asarray([weight for *_, weight in space.graph.edges(data='weight', default=1.0)], dtype=float)
            return np.concatenate([first, second]), np.concatenate([second, first]), np.concatenate([weights, weights])
        neighborhood = BallNeighborhood(metric='hexagonal') if space.geometry == 'triangular' else BallNeighborhood()
        coordinates = np.indices(space.shape_).reshape(2, -1)
        sites = np.arange(space.size_)
        pairs = [
            (sites, np.ravel_multi_index(tuple(coordinates + np.asarray(offset)[:, None]), space.shape_, mode='wrap'))
            for offset in neighborhood.prepare().offsets_
        ]
        first, second = (np.concatenate(items) for items in zip(*pairs, strict=True))
        return first, second, np.ones(first.size)

    @staticmethod
    def _get_homotypic_contacts(
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

    @staticmethod
    def _get_area_deviation(sizes: Array, target_area: float, t: float, values: Mapping[str, object]) -> float:
        """Return the mean absolute deviation of the cell areas from the target area.

        It is relative to the target area.
        """
        del t
        cells = np.rint(np.ravel(np.asarray(values['tissue.cell'], dtype=float))).astype(np.intp)
        areas = np.bincount(cells, weights=sizes)
        return float(np.abs(areas - target_area).mean() / target_area)

    @staticmethod
    def _get_cellular_potts_observables(system: CompositeSystem) -> list[ObservableSpec]:
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
            ObservableSpec(
                'homotypic_contacts',
                function=partial(
                    CellularPottsSystem._get_homotypic_contacts,
                    CellularPottsSystem._get_contact_pairs(space),
                    kinds,
                ),
            ),
            ObservableSpec(
                'area_deviation',
                function=partial(CellularPottsSystem._get_area_deviation, sizes, float(adhesion.target_area)),
            ),
        ]

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
        settings['observables'] = CellularPottsSystem._get_cellular_potts_observables(self._build_system())
        return settings

    @staticmethod
    def _get_disordered_lattice(shape: tuple[int, int], rng: np.random.Generator) -> nx.Graph:
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

    def _build_system(self) -> CompositeSystem:
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
            graph = self._get_disordered_lattice((rows, columns), rng)
            x, y = np.asarray([point for _, point in graph.nodes(data='pos')]).T
            space: LatticeSpace | NetworkSpace = NetworkSpace(graph)
        else:
            y, x = np.indices((rows, columns)).reshape(2, -1).astype(float)
            space = LatticeSpace((rows, columns), geometry=self.lattice)
        cells = (y // height).astype(int) * (columns // width) + (x // width).astype(int)
        tissue = GroupElement(
            'tissue',
            element=TissueSiteElement,
            space=space,
            values={'cell': cells.astype(float)},
            on='sites',
        )
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
        return CompositeSystem(self.label, elements=[tissue], interactions=[sampling], time=DiscreteTime())
