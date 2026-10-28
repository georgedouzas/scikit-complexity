"""Define the agent state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ._agent_snapshot_spec import AgentSnapshotSpec
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class AgentStateSpec(BaseSpec):
    """Agents of every group and the model reporters at one frame.

    Args:
        groups:
            A mapping from group path to the agents of that group.

        reporters:
            A mapping from dotted reporter name to its value.
    """

    groups: Mapping[str, AgentSnapshotSpec] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
