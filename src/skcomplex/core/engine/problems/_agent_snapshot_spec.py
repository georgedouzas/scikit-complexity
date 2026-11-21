"""Define the agent snapshot spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...base import BaseSpec
from ...types import AnyArray, IntArray


@dataclass(frozen=True)
class AgentSnapshotSpec(BaseSpec):
    """A spec of the agents of one group alive at one frame.

    It holds the ids of the agents, such as `[1, 2, 3, 4, 5]` for five traders, their values by name, their places, and
    the members their roles name. `Trajectory.get_agents` returns one, for the traders of a market, the birds of a
    flock, and the residents of a city.

    Read more in the user guide, under Agents.

    Args:
        ids:
            The unique id of every agent.

        values:
            A mapping from state name to one value per agent.

        locations:
            One row per agent: grid coordinates, a network node, or a continuous position, or `None` without a space.

        roles:
            A mapping from the name of a state with a role to its role, `'position'`, `'direction'`, or `'velocity'`.

    See Also:
        Trajectory: Returns it from `get_agents`.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> trajectory = BoltzmannWealthSystem(n_agents=5).prepare().simulate(until=1)
        >>> trajectory.get_agents('traders', 1).ids.tolist()
        [1, 2, 3, 4, 5]
    """

    ids: IntArray
    values: Mapping[str, AnyArray]
    locations: AnyArray | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
