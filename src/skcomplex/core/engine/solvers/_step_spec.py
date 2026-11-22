"""Define the step spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...types import Array


@dataclass(frozen=True)
class StepSpec(BaseSpec):
    """A spec of one step of a run, a state at a time and the time to move it to.

    It holds the time `t`, the state `x`, and the parameters `p` at the start of a step, and the time `until` a solver
    moves the state to, such as from 0 to 1. A solver reads it in `run`. It carries a step of a population, a magnet,
    and a market alike.

    Read more in the user guide, under Solvers.

    Args:
        t:
            The current time.

        x:
            The current state vector.

        p:
            The parameter vector.

        until:
            The time to advance to.

    See Also:
        BaseSolver: Moves the state of a step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.solvers import StepSpec
        >>> StepSpec(0.0, np.ones(1), np.zeros(0), 1.0).until
        1.0
    """

    t: float
    x: Array
    p: Array
    until: float
