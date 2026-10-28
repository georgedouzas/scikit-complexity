"""Step of a run: the state at one time and the time a solver advances it to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...core.types import Vector
from ..problems import BaseValue


@dataclass(frozen=True)
class Step(BaseValue):
    """Step of a run: the state at one time and the time a solver advances it to.

    Args:
        t:
            The current time.

        x:
            The current state vector.

        p:
            The parameter vector.

        until:
            The time to advance to.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.solvers import Step
        >>> Step(0.0, np.ones(1), np.zeros(0), 1.0).until
        1.0
    """

    t: float
    x: Vector
    p: Vector
    until: float
