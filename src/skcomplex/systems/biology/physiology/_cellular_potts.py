"""Cellular Potts model of a confluent tissue on a square, triangular, or disordered lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi

from ....core.components.elements import LatticeElement
from ....core.components.spaces import (
    DiscreteTime,
    LatticeSpace,
    MooreNeighborhood,
    NetworkSpace,
    TriangularNeighborhood,
)
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec
from ....core.types import Array, Indices
from ..._catalog import register_system
from ._cell_adhesion import CellAdhesionInteraction
from ._tissue_site import TissueSiteElement

LATTICES = ('square', 'triangular', 'disordered')
DEFAULT_ADHESION = ((1.0, 2.0), (2.0, 1.5))
TILES = 3


def _disordered_lattice(shape: tuple[int, int], rng: np.random.Generator) -> nx.Graph:
    """Return the periodic Voronoi tessellation of jittered points as a graph of sites, one point per unit area.

    Every node has the position `'pos'` of its point and the area `'volume'` of its Voronoi cell, and every edge joins
    two cells that share a side, with the length of the side over the mean length as its `'weight'`.
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


def _contact_pairs(space: LatticeSpace | NetworkSpace) -> tuple[Indices, Indices, Array]:
    """Return the two sites and the weight of every contact of the tissue, each contact in both directions."""
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
    """Return the share of the contact between different cells that joins cells of the same kind."""
    del t
    first, second, weights = pairs
    cells = np.rint(np.ravel(np.asarray(values['tissue.cell'], dtype=float))).astype(np.intp)
    boundary = cells[first] != cells[second]
    same = kinds[cells[first]] == kinds[cells[second]]
    return float((weights * boundary * same).sum() / max((weights * boundary).sum(), np.finfo(float).tiny))


def _area_deviation(sizes: Array, target_area: float, t: float, values: Mapping[str, object]) -> float:
    """Return the mean absolute deviation of the cell areas from the target area, relative to the target area."""
    del t
    cells = np.rint(np.ravel(np.asarray(values['tissue.cell'], dtype=float))).astype(np.intp)
    areas = np.bincount(cells, weights=sizes)
    return float(np.abs(areas - target_area).mean() / target_area)


def _cellular_potts_observables(system: System) -> list[ObservableSpec]:
    """Return the share of homotypic contacts and the deviation of the areas from their target."""
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


@register_system(
    field='biology/physiology',
    tags=('cellular potts', 'tissue', 'cell sorting', 'differential adhesion', 'lattice', 'monte carlo'),
    references=('Nemati and de Graaf 2024, arXiv 2404.09055', 'Graner and Glazier 1992'),
    defaults={'until': 2000, 'dt': 1},
    observables=_cellular_potts_observables,
)
def cellular_potts(
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
) -> System:
    """Return the cellular Potts model of a confluent tissue whose cells of several kinds sort by differential adhesion.

    Every site of a periodic lattice belongs to one cell, and the cells start as rectangles of `cell_shape` sites, whose
    area is the target area, with kinds in equal shares at random. Label copies sample the energy
    `sum_<i,j> w_ij J(t_i, t_j) [s_i != s_j] + lambda sum_s (a_s - A)^2 + mu sum_s (p_s - P)^2` of Nemati and de Graaf
    (2024), Eq. 3, extended to the adhesion `J` between kinds of Graner and Glazier (1992). The square lattice reads the
    Moore neighbourhood and the triangular lattice the six nearest neighbours, both with unit weights, and the
    disordered lattice is the periodic Voronoi tessellation of one jittered point per unit area, whose sites have the
    area of their Voronoi cell and touch along sides of weight `w_ij`, the length of the side over the mean length. A
    heterotypic adhesion above the mean of the homotypic ones raises the share of contacts between cells of one kind.

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

    Returns:
        system:
            The system, with the states `tissue.cell`.

    Raises:
        ValidationError:
            The lattice is unknown, or the shape is not a multiple of the cell shape.

    Examples:
        >>> from skcomplex.systems.biology.physiology import cellular_potts
        >>> system = cellular_potts((10, 16))
        >>> system.interactions[0].kinds
        (0, 0, 1, 1)
        >>> cellular_potts((10, 16), lattice='disordered').elements[0].space.graph.number_of_nodes()
        160
    """
    rows, columns = (int(size) for size in shape)
    height, width = (int(size) for size in cell_shape)
    if lattice not in LATTICES:
        error_msg = f'The lattice of the cellular Potts model is one of {", ".join(LATTICES)}, but got {lattice!r}.'
        raise ValidationError(error_msg)
    if rows % height or columns % width:
        error_msg = f'The shape {(rows, columns)} of the tissue is not a multiple of the cell shape {(height, width)}.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    count = (rows // height) * (columns // width)
    kinds = np.asarray(adhesion, dtype=float).shape[0]
    kind_of = tuple(int(kind) for kind in rng.permutation(np.arange(count) % kinds))
    if lattice == 'disordered':
        graph = _disordered_lattice((rows, columns), rng)
        x, y = np.asarray([point for _, point in graph.nodes(data='pos')]).T
        space: LatticeSpace | NetworkSpace = NetworkSpace(graph)
    else:
        y, x = np.indices((rows, columns)).reshape(2, -1).astype(float)
        space = LatticeSpace((rows, columns), geometry=lattice)
    cells = (y // height).astype(int) * (columns // width) + (x // width).astype(int)
    tissue = LatticeElement('tissue', element=TissueSiteElement, space=space, values={'cell': cells.astype(float)})
    sampling = CellAdhesionInteraction(
        'adhesion',
        target='tissue',
        kinds=kind_of,
        adhesion=adhesion,
        temperature=temperature,
        area_strength=area_strength,
        target_area=float(height * width),
        perimeter_strength=perimeter_strength,
        target_perimeter=target_perimeter,
        hits=hits,
    )
    return System(elements=[tissue], interactions=[sampling], time=DiscreteTime())
