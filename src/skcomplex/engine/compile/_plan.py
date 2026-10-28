"""Partitions of a system and the issues of its law mixes that no form or solver takes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._issue import Issue
from ._planned_partition import _PlannedPartition


@dataclass(frozen=True)
class _Plan:
    """Partitions of a system and the issues of its law mixes that no form or solver takes."""

    partitions: tuple[_PlannedPartition, ...]
    issues: tuple[Issue, ...]
