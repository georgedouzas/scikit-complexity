"""Cooling systems.

A cup of coffee relaxes to the temperature of the room.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.particles import RelaxationTimeInteraction


class CoffeeCupElement(Element):
    """Cup of coffee at the temperature `T` in a room at the temperature `room`.

    The cup exchanges heat at the rate `k`.

    Args:
        label:
            The label of the cup.

        visual:
            How the cup is drawn, or `None` for none.

        T:
            The initial temperature of the coffee.

        room:
            The temperature of the room.

        k:
            The rate of heat exchange.

    Examples:
        >>> from skcomplex.systems.physics.heat import CoffeeCupElement
        >>> CoffeeCupElement('coffee', k=0.2).k
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 90.0,
        room: Annotated[float, ParameterSpec()] = 20.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.room = room
        self.k = k


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
        >>> CoolingCoffeeSystem(T=5.0).prepare().system_.elements[0].T
        5.0
    """

    field = "physics/heat"
    topics = ("heat", "ode", "balancing feedback", "stocks and flows")
    references = ("Newton 1701", "Meadows 2008")
    until = 60.0
    dt = 0.5

    def __init__(
        self,
        T: float = 90.0,
        *,
        room: float = 20.0,
        k: float = 0.1,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.T = T
        self.room = room
        self.k = k

    def _build_system(self) -> System:
        """Return a cup of coffee whose temperature relaxes to that of the room.

        The temperature follows `T' = -k (T - T_room)`.
        """
        cooling = RelaxationTimeInteraction(
            "cooling",
            target="coffee",
            state="T",
            goal=lambda s: s.target.room,
            time=lambda s: 1 / s.target.k,
        )
        return System(
            self.label,
            elements=[CoffeeCupElement("coffee", T=self.T, room=self.room, k=self.k)],
            interactions=[cooling],
        )
