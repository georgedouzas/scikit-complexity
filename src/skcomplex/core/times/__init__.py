"""Define the time of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseTime
from ._continuous_time import ContinuousTime
from ._discrete_time import DiscreteTime
from ._static_time import StaticTime

__all__: list[str] = [
    'BaseTime',
    'ContinuousTime',
    'DiscreteTime',
    'StaticTime',
]
