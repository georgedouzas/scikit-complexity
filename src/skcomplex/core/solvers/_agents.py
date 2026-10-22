"""Record the agents of agent-based problems at every frame, whichever backend runs them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AgentSnapshot:
    """Agents of one group alive at one frame.

    Args:
        ids:
            The unique id of every agent.

        values:
            A mapping from state name to one value per agent.

        locations:
            One row per agent: grid coordinates, a network node, or a continuous position, or `None` without a space.
    """

    ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    locations: npt.NDArray[Any] | None = None

    def __len__(self) -> int:
        """Return the number of agents."""
        return len(self.ids)


@dataclass(frozen=True)
class AgentState:
    """Agents of every group and the model reporters at one frame.

    Args:
        groups:
            A mapping from group path to the agents of that group.

        reporters:
            A mapping from dotted reporter name to its value.
    """

    groups: Mapping[str, AgentSnapshot] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)
