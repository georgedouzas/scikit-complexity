"""Define the control spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _ControlSpec(BaseSpec):
    """Control of an optimal control problem: its dotted name, state index, symbol, and bounds."""

    name: str
    index: int
    symbol: Expression
    lower: float
    upper: float
