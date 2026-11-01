"""Define the output spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._noise_process_spec import NoiseProcessSpec
from ...types import Indices
from ..types import NumericFunction
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class OutputSpec(BaseSpec):
    """Value a kernel produces for one target component.

    Args:
        target:
            The target component, or `None` for the propensity of a jump.

        expression:
            The symbolic value, or `None` when a callable produces it.

        function:
            The callable or `Call` function that produces the values of every component of its target, or `None`.

        arguments:
            The argument expressions of a `Call` function, or `None` for a callable law.

        component:
            The component of the callable's result that this output holds.

        components:
            The number of components the callable returns per evaluation index.

        process:
            The named Wiener process of a noise output, or `None` for one independent process per evaluation index.

        combined:
            Whether an assigned output adds its change from the state to the value an earlier output set, so that
            several update laws of one state combine.

        reduce:
            `'sum'` or `'mean'` for an assigned output that sets every member it writes to the sum or the mean of the
            values of the rows that write it, or `None`.

        counts:
            The number of rows, over every kernel, that write the member of every row of an output that reduces, or
            `None`.
    """

    target: ReferenceSpec | None
    expression: Expression | None = None
    function: NumericFunction | Callable[..., object] | None = None
    arguments: tuple[Expression, ...] | None = None
    component: int = 0
    components: int = 1
    process: NoiseProcessSpec | None = None
    combined: bool = False
    reduce: str | None = None
    counts: Indices | None = field(default=None, compare=False, repr=False)
