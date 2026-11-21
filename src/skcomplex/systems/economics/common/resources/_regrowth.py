"""Define regrowth, the linear growth back of a renewable resource.

It grows back up to its capacity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import UpdateLaw
from .....core.components.types import SystemViewProtocol
from .....core.types import Role


class RegrowthInteraction(Interaction):
    """Linear regrowth of a resource held by the agents of a group.

    It grows by a fixed amount a step up to their capacity. Every step the resource `x` of an agent grows by the amount
    `g` its parameter `growth` names, `x = min(K, x + g)`, or by that share of its capacity `K` when the regrowth is
    relative, `x = min(K, x + g K)`. It is a rule of the system that grows every agent, such as the sugar of a patch of
    Sugarscape or the food of a plot of land.

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
        >>> regrowth.growth
        'growback'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str,
        capacity: str = 'capacity',
        growth: str = 'growth',
        relative: bool = False,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.capacity = capacity
        self.growth = growth
        self.relative = relative
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the regrowth of every agent of the target as a rule of the system, in its phase.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self._regrow, phase=self.phase)]

    def _regrow(self, system: SystemViewProtocol) -> None:
        """Grow the resource of every agent back, at most to its capacity."""
        for agent in system.get_agents('target'):
            capacity = float(getattr(agent, self.capacity))
            growth = float(getattr(agent, self.growth)) * (capacity if self.relative else 1.0)
            setattr(agent, self.state, min(capacity, float(getattr(agent, self.state)) + growth))
