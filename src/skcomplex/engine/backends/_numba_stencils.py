"""Read fields at neighbouring lattice sites inside compiled kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from numba import njit

PERIODIC, ZERO_FLUX = 0, 1


@njit(cache=True)
def _neighbor_value(
    values: npt.NDArray[np.float64],
    member: int,
    read: tuple[int, int, int],
    offset: tuple[tuple[int, ...], tuple[int, ...]],
    edges: tuple[int, float],
) -> float:
    """Return a field component at the site at an offset from a member's site of a row-major lattice.

    Args:
        values:
            The state or parameter vector.

        member:
            The member, the site in row-major order.

        read:
            The index of the first component of the field, the number of components per member, and the component to
            read.

        offset:
            The lattice shape and the offset along every dimension.

        edges:
            The edge mode, 0 for periodic edges, 1 for zero-flux edges that mirror the lattice, and 2 for fixed edges,
            and the value read outside fixed edges.

    Returns:
        value:
            The component at the neighbouring site.
    """
    start, columns, column = read
    sizes, steps = offset
    mode, ghost = edges
    remainder = np.int64(member)
    index = np.int64(0)
    stride = 1
    for axis in range(len(sizes) - 1, -1, -1):
        size = sizes[axis]
        position = remainder % size
        remainder //= size
        shifted = position + steps[axis]
        if mode == PERIODIC:
            shifted %= size
        elif mode == ZERO_FLUX:
            shifted %= 2 * size
            if shifted >= size:
                shifted = 2 * size - 1 - shifted
        elif shifted < 0 or shifted >= size:
            return ghost
        index += shifted * stride
        stride *= size
    return values[start + index * columns + column]
