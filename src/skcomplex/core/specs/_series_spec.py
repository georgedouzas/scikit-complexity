"""Define the series spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SeriesSpec(BaseSpec):
    """Named series on common times.

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
