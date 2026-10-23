"""Find the neighbours of members within a radius, among the nearest, or inside a cone with cell lists in Numba."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import numpy.typing as npt
from numba import njit

type Floats = npt.NDArray[np.float64]
type Integers = npt.NDArray[np.int64]
type Shape = tuple[Integers, Floats, Floats]
type Grid = tuple[Shape, Integers, Integers, Integers]
type Box = tuple[Floats, bool]
type Aim = tuple[Floats, float, bool]
type Context = tuple[Grid, Aim, Box]
type Found = tuple[Integers, Floats, bool]

CELLS_PER_MEMBER = 8
FEWEST_CELLS = 64


@njit(cache=True)
def _coordinate(value: float, dimension: int, shape: Shape, box: Box) -> int:
    """Return the cell coordinate of a position component, inside the grid of a shape of counts, sides, and origin."""
    counts, sides, origin = shape
    periods, periodic = box
    if periodic:
        wrapped = value % periods[dimension]
        return min(int(wrapped // sides[dimension]), counts[dimension] - 1)
    return max(0, min(int((value - origin[dimension]) // sides[dimension]), counts[dimension] - 1))


@njit(cache=True)
def _shape(side: float, low: Floats, high: Floats, box: Box) -> Shape:
    """Return the cell counts, sides, and origin of a grid of cells of at least a side over a box."""
    periods, periodic = box
    n_dim = low.size
    counts = np.ones(n_dim, dtype=np.int64)
    sides = np.full(n_dim, side)
    for dimension in range(n_dim):
        if periodic:
            counts[dimension] = max(1, int(periods[dimension] // side))
            sides[dimension] = periods[dimension] / counts[dimension]
        else:
            counts[dimension] = int((high[dimension] - low[dimension]) // side) + 1
    return counts, sides, low


@njit(cache=True)
def _grid(positions: Floats, side: float, box: Box) -> Grid:
    """Return a grid of cells of at least a side, coarsened to a few cells per member, with the members binned.

    The grid is the shape of counts, sides, and origin, the strides, the start of every cell in the order, and the
    members in cell order.
    """
    n_members, n_dim = positions.shape
    low, high = np.zeros(n_dim), np.zeros(n_dim)
    if not box[1] and n_members > 0:
        for dimension in range(n_dim):
            low[dimension] = positions[:, dimension].min()
            high[dimension] = positions[:, dimension].max()
    limit = max(FEWEST_CELLS, CELLS_PER_MEMBER * n_members)
    shape = _shape(side, low, high, box)
    while np.prod(shape[0].astype(np.float64)) > limit:
        side *= 2.0
        shape = _shape(side, low, high, box)
    counts = shape[0]
    strides = np.ones(n_dim, dtype=np.int64)
    for dimension in range(1, n_dim):
        strides[dimension] = strides[dimension - 1] * counts[dimension - 1]
    n_cells = strides[n_dim - 1] * counts[n_dim - 1]
    cells = np.empty(n_members, dtype=np.int64)
    starts = np.zeros(n_cells + 1, dtype=np.int64)
    for member in range(n_members):
        flat = 0
        for dimension in range(n_dim):
            flat += _coordinate(positions[member, dimension], dimension, shape, box) * strides[dimension]
        cells[member] = flat
        starts[flat + 1] += 1
    for cell in range(n_cells):
        starts[cell + 1] += starts[cell]
    filled = starts[:-1].copy()
    order = np.empty(n_members, dtype=np.int64)
    for member in range(n_members):
        order[filled[cells[member]]] = member
        filled[cells[member]] += 1
    return shape, strides, starts, order


@njit(cache=True)
def _inside(positions: Floats, member: int, other: int, limit: float, context: Context) -> tuple[bool, float]:
    """Return whether a member is a neighbour of another within a squared distance and a cone, and its distance."""
    _, aim, box = context
    headings, cosine, cone = aim
    periods, periodic = box
    squared, dot, heading = 0.0, 0.0, 0.0
    for dimension in range(positions.shape[1]):
        difference = positions[member, dimension] - positions[other, dimension]
        if periodic:
            difference -= periods[dimension] * np.rint(difference / periods[dimension])
        squared += difference * difference
        if cone:
            dot += headings[member, dimension] * -difference
            heading += headings[member, dimension] * headings[member, dimension]
    inside = squared <= limit and (not cone or dot >= cosine * (math.sqrt(heading) * math.sqrt(squared)))
    return inside, squared


@njit(cache=True)
def _cells_around(positions: Floats, member: int, reach: float, context: Context) -> tuple[Integers, Integers]:
    """Return the lowest and highest cell coordinates of the cells within a reach of a member, each cell once."""
    grid, _, box = context
    shape = grid[0]
    counts, sides, _ = shape
    n_dim = positions.shape[1]
    low = np.empty(n_dim, dtype=np.int64)
    high = np.empty(n_dim, dtype=np.int64)
    for dimension in range(n_dim):
        center = _coordinate(positions[member, dimension], dimension, shape, box)
        steps = math.ceil(reach / sides[dimension])
        if box[1] and 2 * steps + 1 >= counts[dimension]:
            low[dimension], high[dimension] = 0, counts[dimension] - 1
        elif box[1]:
            low[dimension], high[dimension] = center - steps, center + steps
        else:
            low[dimension] = max(0, center - steps)
            high[dimension] = min(counts[dimension] - 1, center + steps)
    return low, high


@njit(cache=True)
def _next_cell(current: Integers, low: Integers, high: Integers) -> bool:
    """Advance the coordinates of a cell to the next cell of a block, and return whether the block is done."""
    for dimension in range(current.size):
        current[dimension] += 1
        if current[dimension] <= high[dimension]:
            return False
        current[dimension] = low[dimension]
    return True


@njit(cache=True)
def _scan(positions: Floats, member: int, reach: float, context: Context, found: Found) -> int:
    """Return the number of other members within a reach and a cone of a member, storing them and their distances."""
    grid = context[0]
    counts, strides, starts, order = grid[0][0], grid[1], grid[2], grid[3]
    indices, distances, store = found
    low, high = _cells_around(positions, member, reach, context)
    current = low.copy()
    limit = reach * reach
    size = 0
    done = False
    while not done:
        flat = 0
        for dimension in range(current.size):
            flat += (current[dimension] % counts[dimension]) * strides[dimension]
        for slot in range(starts[flat], starts[flat + 1]):
            other = order[slot]
            if other == member:
                continue
            inside, squared = _inside(positions, member, other, limit, context)
            if inside and store:
                indices[size] = other
                distances[size] = squared
            size += 1 if inside else 0
        done = _next_cell(current, low, high)
    return size


@njit(cache=True)
def _within(positions: Floats, radius: float, aim: Aim, box: Box, center: bool) -> tuple[Integers, Integers]:
    """Return the members within a radius, and inside a cone, of every member as CSR rows sorted by member.

    The search is serial: a kernel loaded from the Numba cache crashes when it calls a parallel function through a
    cached one, and generated kernels call the searches that way.
    """
    n_members = positions.shape[0]
    context = (_grid(positions, radius, box), aim, box)
    counts = np.zeros(n_members, dtype=np.int64)
    for member in range(n_members):
        counts[member] = _scan(positions, member, radius, context, (np.empty(0, dtype=np.int64), np.empty(0), False))
    indptr = np.zeros(n_members + 1, dtype=np.int64)
    for member in range(n_members):
        indptr[member + 1] = indptr[member] + counts[member] + (1 if center else 0)
    indices = np.empty(indptr[n_members], dtype=np.int64)
    for member in range(n_members):
        start, end = indptr[member], indptr[member + 1]
        _scan(positions, member, radius, context, (indices[start:end], np.empty(end - start), True))
        if center:
            indices[end - 1] = member
        indices[start:end] = np.sort(indices[start:end])
    return indptr, indices


@njit(cache=True)
def radius_neighbours(
    positions: Floats,
    radius: float,
    periods: Floats,
    periodic: bool,
    center: bool,
) -> tuple[Integers, Integers]:
    """Return the members within a radius of every member, as CSR rows sorted by member.

    Args:
        positions:
            One row of position components per member.

        radius:
            The largest distance of a neighbour.

        periods:
            The sides of the periodic box, read when it is periodic.

        periodic:
            Whether distances are minimum images in the box.

        center:
            Whether every member is its own neighbour.

    Returns:
        rows:
            The start of the neighbours of every member, and the neighbours.
    """
    return _within(positions, radius, (np.zeros((1, 1)), 1.0, False), (periods, periodic), center)


@njit(cache=True)
def cone_neighbours(  # noqa: PLR0917
    positions: Floats,
    headings: Floats,
    radius: float,
    cosine: float,
    box: Box,
    center: bool,
) -> tuple[Integers, Integers]:
    """Return the members within a radius and inside the cone around the heading of every member, as CSR rows.

    Args:
        positions:
            One row of position components per member.

        headings:
            One row of heading components per member.

        radius:
            The largest distance of a neighbour.

        cosine:
            The cosine of the half-angle of the cone.

        box:
            The sides of the periodic box and whether distances are minimum images in it.

        center:
            Whether every member is its own neighbour.

    Returns:
        rows:
            The start of the neighbours of every member, and the neighbours.
    """
    return _within(positions, radius, (headings, cosine, True), box, center)


@njit(cache=True)
def _side(positions: Floats, k: int, box: Box) -> float:
    """Return the side of cells that hold about `k + 1` members each."""
    n_members, n_dim = positions.shape
    periods, periodic = box
    volume = 1.0
    for dimension in range(n_dim):
        if periodic:
            volume *= periods[dimension]
        elif n_members > 0:
            volume *= max(positions[:, dimension].max() - positions[:, dimension].min(), 1e-12)
    side = (volume * (k + 1) / max(n_members, 1)) ** (1.0 / n_dim)
    return side if side > 0 else 1.0


@njit(cache=True)
def _nearest_of(positions: Floats, member: int, k: int, side: float, context: Context) -> Integers:
    """Return the `k` nearest other members of a member, doubling the reach of the search until it holds them."""
    reach = side
    nothing = (np.empty(0, dtype=np.int64), np.empty(0), False)
    size = _scan(positions, member, reach, context, nothing)
    while size < k:
        reach *= 2.0
        size = _scan(positions, member, reach, context, nothing)
    candidates, distances = np.empty(size, dtype=np.int64), np.empty(size)
    _scan(positions, member, reach, context, (candidates, distances, True))
    return candidates[np.argsort(distances, kind='mergesort')[:k]]


@njit(cache=True)
def nearest_neighbours(
    positions: Floats,
    k: int,
    periods: Floats,
    periodic: bool,
    center: bool,
) -> tuple[Integers, Integers]:
    """Return the `k` nearest other members of every member, as CSR rows sorted by member.

    The search starts in the cells of a grid sized for about `k` members per cell and doubles its reach until it
    holds `k` members, every member outside the cells it searched being farther than its reach. The search is serial,
    as in `radius_neighbours`.

    Args:
        positions:
            One row of position components per member.

        k:
            The number of neighbours.

        periods:
            The sides of the periodic box, read when it is periodic.

        periodic:
            Whether distances are minimum images in the box.

        center:
            Whether every member is its own neighbour, besides its `k` nearest others.

    Returns:
        rows:
            The start of the neighbours of every member, and the neighbours.
    """
    n_members = positions.shape[0]
    k = max(0, min(k, n_members - 1))
    box = (periods, periodic)
    side = _side(positions, k, box)
    context = (_grid(positions, side, box), (np.zeros((1, 1)), 1.0, False), box)
    extra = 1 if center else 0
    indptr = np.arange(n_members + 1, dtype=np.int64) * (k + extra)
    indices = np.empty(indptr[n_members], dtype=np.int64)
    for member in range(n_members):
        start, end = indptr[member], indptr[member + 1]
        if k > 0:
            indices[start : start + k] = _nearest_of(positions, member, k, side, context)
        if center:
            indices[end - 1] = member
        indices[start:end] = np.sort(indices[start:end])
    return indptr, indices


@njit(cache=True)
def _edge_counts(
    p: Floats,
    ends: tuple[int, int, int],
    n_edges: int,
    n_nodes: int,
    flags: tuple[bool, bool],
) -> Integers:
    """Return the start of the neighbours of every node along the active edge rows, and their number last."""
    sources, targets, actives = ends
    directed, center = flags
    counts = np.zeros(n_nodes + 1, dtype=np.int64)
    for edge in range(n_edges):
        if p[actives + edge] > 0.5:  # noqa: PLR2004
            counts[int(p[sources + edge]) + 1] += 1
            counts[int(p[targets + edge]) + 1] += 0 if directed else 1
    for node in range(n_nodes if center else 0):
        counts[node + 1] += 1
    return np.cumsum(counts)


@njit(cache=True)
def _sorted_rows(indptr: Integers, indices: Integers, edges: Integers) -> None:
    """Sort the neighbours of every node, keeping their edges aligned."""
    for node in range(indptr.size - 1):
        start, end = indptr[node], indptr[node + 1]
        order = np.argsort(indices[start:end], kind='mergesort')
        indices[start:end] = indices[start:end][order]
        edges[start:end] = edges[start:end][order]


@njit(cache=True)
def edge_neighbours(  # noqa: PLR0917
    p: Floats,
    ends: tuple[int, int, int],
    n_edges: int,
    n_nodes: int,
    directed: bool,
    center: bool,
) -> tuple[Integers, Integers, Integers]:
    """Return the neighbours of every node along the active edge rows of a layer, with the edge row of every one.

    The edge rows hold their source, target, and activity in the parameter vector from the starts `ends`.
    """
    sources, targets, actives = ends
    indptr = _edge_counts(p, ends, n_edges, n_nodes, (directed, center))
    filled = indptr[:-1].copy()
    indices = np.empty(indptr[-1], dtype=np.int64)
    edges = np.empty(indptr[-1], dtype=np.int64)
    for node in range(n_nodes if center else 0):
        indices[filled[node]], edges[filled[node]] = node, -1
        filled[node] += 1
    for edge in range(n_edges):
        if p[actives + edge] <= 0.5:  # noqa: PLR2004
            continue
        first, second = int(p[sources + edge]), int(p[targets + edge])
        indices[filled[first]], edges[filled[first]] = second, edge
        filled[first] += 1
        if not directed:
            indices[filled[second]], edges[filled[second]] = first, edge
            filled[second] += 1
    _sorted_rows(indptr, indices, edges)
    return indptr, indices, edges
