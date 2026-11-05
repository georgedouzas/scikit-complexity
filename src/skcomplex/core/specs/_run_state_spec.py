"""Define the run state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RunStateSpec(BaseSpec):
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

        source:
            The values of the node a topology request starts from by dotted name, or `None` for a kernel of no request.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import RunStateSpec
        >>> RunStateSpec(0.0, np.ones(2), np.zeros(0)).x.tolist()
        [1.0, 1.0]
    """

    t: float
    x: Array
    p: Array
    rng: np.random.Generator | None = None
    source: Mapping[str, Array] | None = None
