"""Define the neighbor reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class NeighborReductionSpec(BaseSpec):
    """Neighbourhood reduction found in an expression, with its kind, argument, weight, and neighbourhood.

    Args:
        call:
            The unevaluated call.

        kind:
            One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, `'argmax'`, and the components
            reductions `'component-label'`, `'component-size'`, `'component-largest'`, and `'component-giant'`.

        argument:
            The value of one neighbour.

        weight:
            The weight of one neighbour, or `None` for 1.

        over:
            The neighbourhood, or `None` for the neighbours of the space.
    """

    call: Expression
    kind: str
    argument: Expression
    weight: Expression | None
    over: object
