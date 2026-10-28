"""Value a kernel produces for one target component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sage.all__sagemath_symbolics import Expression

from ._base import BaseValue
from ._noise_process import NoiseProcess
from ._reference import Reference

type NumericFunction = Callable[[float, Mapping[str, object]], object]


@dataclass(frozen=True)
class Output(BaseValue):
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
    """

    target: Reference | None
    expression: Expression | None = None
    function: NumericFunction | Callable[..., Any] | None = None
    arguments: tuple[Expression, ...] | None = None
    component: int = 0
    components: int = 1
    process: NoiseProcess | None = None
