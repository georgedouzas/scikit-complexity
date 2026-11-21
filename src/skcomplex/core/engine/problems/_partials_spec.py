"""Define the partials spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec


@dataclass(frozen=True, eq=False)
class PartialsSpec(BaseSpec):
    """A spec of the nonzero partial derivatives of expressions, compiled together.

    The engine differentiates the rates of a system by every value they read, keeps only the derivatives that are not
    zero, such as the derivative of `x * y` by `x`, and compiles them into one NumPy function, which the Jacobian kernel
    calls. It serves the stability of an ecosystem, stiff chemistry, and the control of a grid.

    Read more in the user guide, under Extend the library.

    Args:
        pairs:
            The index of the expression and the symbol of every derivative.

        function:
            The function of the time and the arguments that returns every derivative, or `None` for no derivative.

    See Also:
        JacobianKernel: Reads the derivatives.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.engine.problems._partials_spec import PartialsSpec
        >>> PartialsSpec(((0, symbol('x')),), None).pairs
        ((0, x),)
    """

    pairs: tuple[tuple[int, Expression], ...]
    function: Callable[..., object] | None
