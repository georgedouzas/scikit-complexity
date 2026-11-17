"""Cooling systems.

A cup of coffee relaxes to the temperature of the room.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class CoffeeCupElement(Element):
    """Cup of coffee at the temperature `T`.

    Args:
        label:
            The label of the cup.

        T:
            The initial temperature of the coffee.

    Examples:
        >>> from skcomplex.systems.physics.heat import CoffeeCupElement
        >>> CoffeeCupElement('coffee', T=60.0).T
        60.0
    """

    def __init__(self, label: str, *, T: float = 90.0) -> None:
        super().__init__(label)
        self.T = T


class NewtonCoolingInteraction(Interaction):
    """Newton cooling of a body towards the temperature of the room.

    The temperature `T` of the target follows `T' = -k (T - room)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the body that cools.

        room:
            The temperature of the room.

        k:
            The rate of heat exchange.

    Examples:
        >>> from skcomplex.systems.physics.heat import NewtonCoolingInteraction
        >>> NewtonCoolingInteraction('cooling', target='coffee', k=0.2).k
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        room: float = 20.0,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.room = room
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the temperature of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T = context.target.T
            self.laws_ = [RateLaw(T.diff(context.t), -context.k * (T - context.room))]


class CoolingCoffeeSystem(BaseSystem):
    """A cup of coffee whose temperature relaxes to that of the room.

    The temperature follows `T' = -k (T - T_room)`.

    The heat flow is proportional to the discrepancy between the coffee and the room, a balancing feedback loop, so hot
    coffee cools and iced coffee warms, both exponentially with the time constant `1 / k`. The system has the
    temperature `coffee.T`.

    Args:
        T:
            The initial temperature of the coffee.

        room:
            The temperature of the room.

        k:
            The rate of heat exchange.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.heat import CoolingCoffeeSystem
        >>> trajectory = CoolingCoffeeSystem(T=5.0).prepare().simulate(until=10.0)
        >>> round(float(trajectory.get_values('coffee.T')[0]), 1)
        5.0
    """

    field = 'physics/heat'
    topics = ('heat', 'ode', 'balancing feedback', 'stocks and flows')
    references = ('Newton 1701', 'Meadows 2008')
    until = 60.0
    dt = 0.5

    def __init__(
        self,
        T: float = 90.0,
        *,
        room: float = 20.0,
        k: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.T = T
        self.room = room
        self.k = k

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a cup of coffee whose temperature relaxes to that of the room.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[CoffeeCupElement('coffee', T=self.T)],
            interactions=[NewtonCoolingInteraction('cooling', target='coffee', room=self.room, k=self.k)],
        )
