"""Define regrowth, the linear growth back of a renewable resource.

It grows back up to its capacity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from .....core.components.interactions import MapInteraction
from .....core.components.types import AgentViewProtocol, SystemViewProtocol
from .....core.specs import RoleSpec


class RegrowthInteraction(MapInteraction):
    """Linear regrowth of a resource held by the agents of a group.

    It grows by a fixed amount a step up to their capacity. Every step the resource `x` of an agent grows by the amount
    `g` its parameter `growth` names, `x = min(K, x + g)`, or by that share of its capacity `K` when the regrowth is
    relative, `x = min(K, x + g K)`. It is a map of the agent form, such as the sugar of a patch of Sugarscape or the
    food of a plot of land.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group whose resource grows back.

        state:
            The name of the state that holds the resource.

        capacity:
            The name of the state or parameter of an agent that holds its capacity.

        growth:
            The name of the parameter of an agent that holds its regrowth in a step.

        relative:
            Whether the regrowth is a share of the capacity rather than an amount.

        phase:
            The phase of the step the regrowth runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.common.resources import RegrowthInteraction
        >>> regrowth = RegrowthInteraction('regrowth', target='patches', state='sugar', growth='growback')
        >>> regrowth.form
        'agent'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        capacity: str = 'capacity',
        growth: str = 'growth',
        relative: bool = False,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, state=state, function=self._get_regrown, form='agent', phase=phase)
        self.capacity = capacity
        self.growth = growth
        self.relative = relative

    def _get_regrown(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the resource of an agent after it grows back, at most its capacity."""
        del system
        capacity = float(getattr(agent, self.capacity))
        growth = float(getattr(agent, self.growth)) * (capacity if self.relative else 1.0)
        return min(capacity, float(getattr(agent, self.state)) + growth)
