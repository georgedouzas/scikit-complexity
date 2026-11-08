"""Define the sampled draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SampledDrawSpec(BaseSpec):
    """Draw from a `scipy.stats` distribution that a stochastic map takes.

    It is taken for every member at every step.

    Args:
        start:
            The index of the first value in the extended parameter vector.

        rows:
            The number of values, one per member.

        sample:
            The function `sample(t, x, q, rng)` that returns the values, given the extended parameter vector `q`.
    """

    start: int
    rows: int
    sample: Callable[[float, Array, Array, np.random.Generator], Array]
