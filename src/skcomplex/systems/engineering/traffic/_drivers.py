"""Driver agents: cars whose drivers choose their speed from the distance to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._careless_braking import CarelessBrakingInteraction
from ._driver import DriverElement
from ._motorist import MotoristElement

ROAD_WIDTH = 1.0


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'agents', 'avalanches', 'stochastic'),
    references=('Charbonneau 2017',),
    defaults={'until': 100},
)
def charbonneau_traffic(
    n_cars: int = 100,
    *,
    probability: float = 0.1,
    safe_distance: float = 5.0,
    speed_limit: float = 10.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Charbonneau's traffic system, where random braking of single drivers sets off traffic jams.

    Cars on a one-way road start at rest with random spacings between 3 and 17. Every step, each driver brakes by 3
    when the car ahead is closer than the safe distance and accelerates by 1 otherwise, within the speed limit. A random
    driver brakes with a probability every step, and then every car moves by its speed.

    Args:
        n_cars:
            The number of cars.

        probability:
            The probability that a random driver brakes in a step.

        safe_distance:
            The distance to the car ahead below which drivers brake.

        speed_limit:
            The largest speed.

        random_state:
            The seed or generator of the initial spacings.

    Returns:
        system:
            The system, with the group `cars` of positions `x`, speeds `v`, and ranks `rank`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import charbonneau_traffic
        >>> charbonneau_traffic(5).elements[0].prepare().size_
        5
    """
    rng = np.random.default_rng(random_state)
    positions = np.cumsum(rng.uniform(3.0, 17.0, n_cars))
    cars = GroupElement(
        'cars',
        element=MotoristElement,
        size=n_cars,
        values={
            'x': positions,
            'v': np.zeros(n_cars),
            'rank': np.arange(n_cars, dtype=float),
            'safe_distance': safe_distance,
            'speed_limit': speed_limit,
        },
    )
    return System(
        elements=[cars],
        interactions=[CarelessBrakingInteraction('braking', cars='cars', probability=probability)],
    )


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'agents', 'stochastic'),
    references=('Downey 2018', 'Resnick 1994'),
    defaults={'until': 100},
)
def ring_road_drivers(
    n_cars: int = 30,
    length: float = 1000.0,
    *,
    eps: float = 0.02,
    speed_limit: float = 40.0,
) -> System:
    """Return drivers on a circular highway, where small random errors of speed set off traffic jams.

    The cars start at rest, equally spaced. Every driver accelerates by 1 per step up to the speed limit, the speed
    takes a random relative error of up to `eps`, and a speed that would reach the car ahead becomes 0.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        eps:
            The relative amplitude of the random error of speed.

        speed_limit:
            The largest speed.

    Returns:
        system:
            The system, with the group `drivers` of speeds `speed` on a ring road of length `length`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import ring_road_drivers
        >>> ring_road_drivers(10).space.prepare().extent_[0]
        1000.0
    """
    positions = np.linspace(0.0, length, n_cars, endpoint=False)
    drivers = GroupElement(
        'drivers',
        element=DriverElement,
        size=n_cars,
        values={'rank': np.arange(n_cars, dtype=float), 'eps': eps, 'speed_limit': speed_limit},
        location=[(float(position), ROAD_WIDTH / 2) for position in positions],
    )
    return System(elements=[drivers], space=ContinuousSpace(extent=(length, ROAD_WIDTH), boundary='periodic'))
