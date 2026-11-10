"""Define the samples spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SamplesSpec(BaseSpec):
    """A spec of the uniformly sampled series that a surrogate is fitted to.

    Args:
        series:
            The values of every series, one row per sample and one column per variable.

        labels:
            The name of every variable.

        dt:
            The time between consecutive samples.
    """

    series: list[Array]
    labels: list[str]
    dt: float
