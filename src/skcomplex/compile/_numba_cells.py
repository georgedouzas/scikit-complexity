"""Bin positions into cells of the cutoff size for compiled neighbour searches."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def _grid(
    positions: npt.NDArray[np.float64],
    cutoff: float,
    period: float,
    periodic: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the number of cells, the cell sides, and the origin of the grid in every dimension."""
    n_members, n_dim = positions.shape
    counts = np.ones(n_dim, dtype=np.int64)
    sides = np.full(n_dim, cutoff)
    origin = np.zeros(n_dim)
    if periodic:
        cells = int(period // cutoff)
        for dimension in range(n_dim):
            counts[dimension] = cells if cells >= 3 else 1  # noqa: PLR2004
            sides[dimension] = period / counts[dimension]
        return counts, sides, origin
    if n_members == 0:
        return counts, sides, origin
    high = positions[0].copy()
    origin[:] = positions[0]
    for member in range(n_members):
        for dimension in range(n_dim):
            value = positions[member, dimension]
            if value < origin[dimension]:
                origin[dimension] = value
            elif value > high[dimension]:
                high[dimension] = value
    for dimension in range(n_dim):
        counts[dimension] = int((high[dimension] - origin[dimension]) // cutoff) + 1
    return counts, sides, origin


@njit(cache=True)
def cell_coordinate(
    value: float,
    dimension: int,
    counts: npt.NDArray[np.int64],
    sides: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    period: float,
    periodic: bool,
) -> int:
    """Return the cell coordinate of a position component, which can fall outside the grid in an open space.

    Args:
        value:
            The position component.

        dimension:
            The dimension of the component.

        counts:
            The number of cells in every dimension.

        sides:
            The side of the cells in every dimension.

        origin:
            The origin of the grid.

        period:
            The side of the periodic box.

        periodic:
            Whether the space is periodic.

    Returns:
        coordinate:
            The cell coordinate.
    """
    if periodic:
        return min(int((value % period) // sides[dimension]), counts[dimension] - 1)
    return int(np.floor((value - origin[dimension]) / sides[dimension]))


@njit(cache=True)
def build_cells(
    positions: npt.NDArray[np.float64],
    cutoff: float,
    period: float,
    periodic: bool,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Sort members into cells whose side is at least the cutoff.

    Args:
        positions:
            One row of position components per member.

        cutoff:
            The interaction cutoff.

        period:
            The side of the periodic box.

        periodic:
            Whether the space is periodic.

    Returns:
        cells:
            The cell counts, sides, origin, strides, the start of every cell in the order, and the members in cell
            order.
    """
    n_members, n_dim = positions.shape
    counts, sides, origin = _grid(positions, cutoff, period, periodic)
    strides = np.ones(n_dim, dtype=np.int64)
    for dimension in range(1, n_dim):
        strides[dimension] = strides[dimension - 1] * counts[dimension - 1]
    n_cells = 1
    for dimension in range(n_dim):
        n_cells *= counts[dimension]
    cells = np.empty(n_members, dtype=np.int64)
    starts = np.zeros(n_cells + 1, dtype=np.int64)
    for member in range(n_members):
        flat = 0
        for dimension in range(n_dim):
            value = (
                positions[member, dimension] % period if periodic else positions[member, dimension] - origin[dimension]
            )
            flat += min(int(value / sides[dimension]), counts[dimension] - 1) * strides[dimension]
        cells[member] = flat
        starts[flat + 1] += 1
    for cell in range(n_cells):
        starts[cell + 1] += starts[cell]
    filled = starts[:-1].copy()
    order = np.empty(n_members, dtype=np.int64)
    for member in range(n_members):
        order[filled[cells[member]]] = member
        filled[cells[member]] += 1
    return counts, sides, origin, strides, starts, order


@njit(cache=True)
def neighbor_cell(
    position: npt.NDArray[np.float64],
    combination: int,
    counts: npt.NDArray[np.int64],
    sides: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    strides: npt.NDArray[np.int64],
    period: float,
    periodic: bool,
) -> int:
    """Return the flat index of one of the `3 ** n_dim` cells around a position, or -1 when it does not exist.

    Args:
        position:
            The position components.

        combination:
            The index of the neighbouring cell, from 0 to `3 ** n_dim - 1`.

        counts:
            The number of cells in every dimension.

        sides:
            The side of the cells in every dimension.

        origin:
            The origin of the grid.

        strides:
            The flat index stride of every dimension.

        period:
            The side of the periodic box.

        periodic:
            Whether the space is periodic.

    Returns:
        cell:
            The flat cell index, or -1.
    """
    flat, rest = 0, combination
    for dimension in range(counts.size):
        offset = rest % 3 - 1
        rest //= 3
        if periodic and counts[dimension] == 1 and offset != 0:
            return -1
        coordinate = cell_coordinate(position[dimension], dimension, counts, sides, origin, period, periodic) + offset
        if periodic:
            coordinate %= counts[dimension]
        elif coordinate < 0 or coordinate >= counts[dimension]:
            return -1
        flat += coordinate * strides[dimension]
    return flat
