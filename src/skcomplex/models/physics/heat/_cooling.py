"""Cooling models: a cup of coffee that relaxes to the temperature of the room."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import NumericDomain, parameter
from ....core.components import ElementComponent
from ....core.laws import RateLaw
from ....core.system import System
from ....engine.runtime import register_model

CoffeeCupComponent = ElementComponent.define(
    'CoffeeCupComponent',
    states={'T': 90.0},
    parameters={'room': 20.0, 'k': parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)},
    laws=lambda s: [RateLaw(s.T, -s.k * (s.T - s.room))],
)


@register_model(
    field='physics/heat',
    tags=('heat', 'ode', 'balancing feedback', 'stocks and flows'),
    references=('Newton 1701', 'Meadows 2008'),
    defaults={'until': 60.0, 'dt': 0.5},
)
def cooling_coffee(T: float = 90.0, *, room: float = 20.0, k: float = 0.1) -> System:
    """Return a cup of coffee whose temperature relaxes to that of the room, `T' = -k (T - T_room)`.

    The heat flow is proportional to the discrepancy between the coffee and the room, a balancing feedback loop, so
    hot coffee cools and iced coffee warms, both exponentially with the time constant `1 / k`.

    Args:
        T:
            The initial temperature of the coffee.

        room:
            The temperature of the room.

        k:
            The rate of heat exchange.

    Returns:
        system:
            The model, with the temperature `coffee.T`.

    Examples:
        >>> from skcomplex.models.physics.heat import cooling_coffee
        >>> cooling_coffee(T=5.0).elements[0].T
        5.0
    """
    return System(components=[CoffeeCupComponent('coffee', T=T, room=room, k=k)])
