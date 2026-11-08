"""Define the plan spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec
from ._issue_spec import IssueSpec
from ._planned_partition_spec import _PlannedPartitionSpec


@dataclass(frozen=True)
class _PlanSpec(BaseSpec):
    """Partitions of a system and the issues of its law mixes.

    The issues are those of the law mixes that no form or solver takes.
    """

    partitions: tuple[_PlannedPartitionSpec, ...]
    issues: tuple[IssueSpec, ...]
