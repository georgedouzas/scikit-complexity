"""Define the agent snapshot spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ..types import AnyArray, IntArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class AgentSnapshotSpec(BaseSpec):
    """Agents of one group alive at one frame.

    Args:
        ids:
            The unique id of every agent.

        values:
            A mapping from state name to one value per agent.

        locations:
            One row per agent: grid coordinates, a network node, or a continuous position, or `None` without a space.
    """

    ids: IntArray
    values: Mapping[str, AnyArray]
    locations: AnyArray | None = None
