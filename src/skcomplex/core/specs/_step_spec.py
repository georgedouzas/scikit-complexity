"""Define the step spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class StepSpec(BaseSpec):
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
        >>> from skcomplex.core.specs import StepSpec
        >>> StepSpec(0.0, np.ones(1), np.zeros(0), 1.0).until
        1.0
    """

    t: float
    x: Array
    p: Array
    until: float
