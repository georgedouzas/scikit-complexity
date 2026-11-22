"""Define the run state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...base import BaseSpec
from ...types import Array


@dataclass(frozen=True)
class RunStateSpec(BaseSpec):
    """A spec of the state of a run at one time, which kernels and searches act at.

    It holds the time `t`, the state `x`, the parameters `p`, the random generator, and the source of outside data, such
    as the size 10 of a population at time 0. Kernels and searches read it in `run`. It carries the state of a
    population, a magnet, and a market alike.

    Read more in the user guide, under Extend the library.

    Args:
        t:
            The time.

        x:
            The state vector.

        p:
            The parameter vector, extended with the reductions or the draws when a kernel reads them.

        rng:
            The random number generator of the run, or `None` for a kernel that draws nothing.

        source:
            The values of the node a topology request starts from by dotted name, or `None` for a kernel of no request.

    See Also:
        BaseKernel: Acts at a run state.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems import RunStateSpec
        >>> RunStateSpec(0.0, np.ones(2), np.zeros(0)).x.tolist()
        [1.0, 1.0]
    """

    t: float
    x: Array
    p: Array
    rng: np.random.Generator | None = None
    source: Mapping[str, Array] | None = None
