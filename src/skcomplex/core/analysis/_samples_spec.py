"""Define the samples spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..specs._base_spec import BaseSpec
from ..types import Array


@dataclass(frozen=True)
class _SamplesSpec(BaseSpec):
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
