"""Define the rate law the compiler keeps with the order of its derivative."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression
from sage.structure.element import Matrix

from ..types import Target
from ._rate import RateLaw


class _OrderedRateLaw(RateLaw):
    """A rate law on a state that keeps the order of the derivative it gives.

    The compiler writes `RateLaw(x.diff(t, k), f)` as this law on `x` of the order `k`, and a rate on a fractional
    derivative as this law of a fractional order. A rate of the order `None` gives the derivative of the order of its
    state.

    Args:
        target:
            The symbol of a scalar state, the column of symbols of a vector state, or the matrix of a tensor state.

        value:
            A symbolic expression, a matrix of the shape of a tensor state, a `Call`, or a callable `f(t, values)`.

        order:
            The order of the derivative the rate gives, or `None` for the order of its state.

        label:
            The name of the rate in messages, or the empty string for none.
    """

    stated_as: ClassVar[str] = 'RateLaw'

    def __init__(
        self,
        target: Target | Matrix | Mapping[Expression, int],
        value: object,
        order: float | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(target, value, label=label)
        self.order = order

    def _get_name(self) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return self.stated_as

    def _get_order(self) -> float | None:
        """Return the order of the derivative the rate gives."""
        return self.order
