"""Define the partials spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec


@dataclass(frozen=True, eq=False)
class PartialsSpec(BaseSpec):
    """Nonzero partial derivatives of expressions by symbols.

    They are compiled into one NumPy function.

    Args:
        pairs:
            The index of the expression and the symbol of every derivative.

        function:
            The function of the time and the arguments that returns every derivative, or `None` for no derivative.
    """

    pairs: tuple[tuple[int, Expression], ...]
    function: Callable[..., object] | None
