"""Define the control problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._control_spec import _ControlSpec
from ...types import Array
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class _ControlProblemSpec(BaseSpec):
    """Optimal control problem of a compiled system.

    Its cost is written to be minimised.

    Args:
        problem:
            The compiled problem.

        names:
            The dotted names of the steered states.

        indices:
            The indices of the steered states in the state vector.

        states:
            The symbols of the steered states.

        controls:
            The controls, the bounded adaptive parameters.

        parameters:
            The symbols of the parameters.

        rates:
            The rates of the steered states.

        running:
            The running cost, written to be minimised.

        terminal:
            The terminal cost, written to be minimised.

        sign:
            1 to minimise the objective, and -1 to maximise it.

        lower:
            The lower bound of every steered state.
    """

    problem: ProblemSpec
    names: tuple[str, ...]
    indices: npt.NDArray[np.intp]
    states: tuple[Expression, ...]
    controls: tuple[_ControlSpec, ...]
    parameters: tuple[Expression, ...]
    rates: tuple[Expression, ...]
    running: Expression
    terminal: Expression
    sign: float
    lower: Array = field(repr=False)

    @property
    def time(self) -> Expression:
        """Time symbol of the expressions."""
        return self.problem.time

    @property
    def control_symbols(self) -> list[Expression]:
        """Symbols of the controls."""
        return [control.symbol for control in self.controls]

    @property
    def control_indices(self) -> npt.NDArray[np.intp]:
        """Indices of the controls in the state vector."""
        return np.array([control.index for control in self.controls], dtype=np.intp)

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Lower and upper bounds of the controls."""
        return (
            np.array([control.lower for control in self.controls]),
            np.array([control.upper for control in self.controls]),
        )
