"""Define the segment spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _SegmentSpec(BaseSpec):
    """Accepted step of an integrator, with its states at both ends and its dense output."""

    start: float
    end: float
    before: Array
    after: Array
    dense: Callable[[float], Array]
