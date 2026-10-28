"""State of a run at one time, which kernels and searches act at."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ...core.types import Vector
from ._base import BaseValue


@dataclass(frozen=True)
class State(BaseValue):
    """State of a run at one time, which kernels and searches act at.

    Args:
        t:
            The time.

        x:
            The state vector.

        p:
            The parameter vector, extended with the reductions or the draws when a kernel reads them.

        rng:
            The random number generator of the run, or `None` for a kernel that draws nothing.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import State
        >>> State(0.0, np.ones(2), np.zeros(0)).x.tolist()
        [1.0, 1.0]
    """

    t: float
    x: Vector
    p: Vector
    rng: np.random.Generator | None = None
