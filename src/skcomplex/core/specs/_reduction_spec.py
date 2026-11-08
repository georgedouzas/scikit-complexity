"""Define the reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..types import AnyArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ReductionSpec(BaseSpec):
    """Reduction call found in an expression.

    It holds its kind, argument, condition, weights, and group.

    Args:
        call:
            The unevaluated call.

        kind:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'var'`, or `'count'`.

        argument:
            The expression of one reduced member, 1 for a count.

        condition:
            The condition that selects the members, or `None` for all members.

        weights:
            The constant weight matrix, the symbol of a shaped parameter, or `None`.

        over:
            A symbol of the group to reduce over, or `None` for the group the argument reads.

        function:
            The function of the array of member values of a reduction of the kind `'function'`, or `None`.
    """

    call: Expression
    kind: str
    argument: Expression
    condition: Expression | None
    weights: object
    over: Expression | None
    function: Callable[[AnyArray], object] | None = None
