"""Define the hyperedge reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class HyperedgeReductionSpec(BaseSpec):
    """Reduction over the hyperedges around every member found in an expression, with its kind, argument, and size.

    Args:
        call:
            The unevaluated call.

        kind:
            One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, and `'argmax'`.

        argument:
            The value of one hyperedge.

        weight:
            The weight of one hyperedge, or `None` for 1.

        size:
            The number of members of the hyperedges reduced over, or `None` for every size.
    """

    call: Expression
    kind: str
    argument: Expression
    weight: Expression | None
    size: int | None
