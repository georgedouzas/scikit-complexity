"""Define the series spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..base import BaseSpec
from ..types import Array


@dataclass(frozen=True)
class _SeriesSpec(BaseSpec):
    """A spec of named series on common times.

    Args:
        times:
            The time of every frame.

        values:
            One row per frame and one column per series.

        labels:
            The name of every column.
    """

    times: Array
    values: Array
    labels: list[str]
