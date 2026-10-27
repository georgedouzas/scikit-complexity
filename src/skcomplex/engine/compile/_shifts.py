"""Define the reading of the arguments of a member offset read, the call `shift_member` returns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import Expression

from ...core.spaces import LatticeSpace


def shift_arguments(call: Expression) -> tuple[Expression, int, str, float]:
    """Return the expression, the offset, the boundary, and the fill of a member offset read.

    Args:
        call:
            A call returned by `shift_member`.

    Returns:
        arguments:
            The expression of member symbols, the number of members along the group order, the boundary, and the
            value read past the ends of a fixed boundary.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.engine.compile._shifts import shift_arguments
        >>> from skcomplex.core.laws import shift_member
        >>> shift_arguments(shift_member(symbol('x'), -2, boundary='fixed', fill=1.0))
        (x, -2, 'fixed', 1.0)
    """
    expression, offset, boundary, fill = call.operands()
    return expression, int(offset), LatticeSpace.boundaries[int(boundary)], float(fill)
