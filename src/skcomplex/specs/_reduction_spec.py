"""Define the reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ReductionSpec(BaseSpec):
    """Reduction call found in an expression, with its kind, argument, condition, weights, and group.

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
    """

    call: Expression
    kind: str
    argument: Expression
    condition: Expression | None
    weights: object
    over: Expression | None
