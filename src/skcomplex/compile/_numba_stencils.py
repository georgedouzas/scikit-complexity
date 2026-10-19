"""Read fields at neighbouring lattice sites inside compiled kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit

PERIODIC, ZERO_FLUX = 0, 1


@njit(cache=True)
def neighbor_value(  # noqa: PLR0917
    values: npt.NDArray[np.float64],
    start: int,
    columns: int,
    column: int,
    member: int,
    sizes: tuple[int, ...],
    steps: tuple[int, ...],
    mode: int,
    ghost: float,
) -> float:
    """Return a field component at the site at an offset from a member's site of a row-major lattice.

    Args:
        values:
            The state or parameter vector.

        start:
            The index of the first component of the field.

        columns:
            The number of components per member.

        column:
            The component to read.

        member:
            The member, the site in row-major order.

        sizes:
            The lattice shape.

        steps:
            The offset along every dimension.

        mode:
            0 for periodic edges, 1 for zero-flux edges that mirror the lattice, and 2 for fixed edges.

        ghost:
            The value read outside fixed edges.

    Returns:
        value:
            The component at the neighbouring site.
    """
    remainder: Any = np.int64(member)
    index = 0
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
