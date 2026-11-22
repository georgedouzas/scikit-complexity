"""Define the agent state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...base import BaseSpec
from ..problems import AgentSnapshotSpec


@dataclass(frozen=True)
class _AgentStateSpec(BaseSpec):
    """A spec of the agents of every group and the reporters at one frame.

    Args:
        groups:
            A mapping from group path to the agents of that group.

        reporters:
            A mapping from dotted reporter name to its value.
    """

    groups: Mapping[str, AgentSnapshotSpec] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
