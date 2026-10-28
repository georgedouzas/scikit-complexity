"""Define the dynamics spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class _DynamicsSpec(BaseSpec):
    """First-order equations of a system: its state symbols, their right sides, and its parameter values."""

    problem: ProblemSpec
    names: list[str]
    symbols: list[Expression]
    rates: list[Expression]
    parameters: dict[Expression, float]
    discrete: bool

    @property
    def time(self) -> Expression:
        """Time symbol of the equations."""
        return self.problem.time
