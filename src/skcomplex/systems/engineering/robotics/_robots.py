"""Robot systems: autonomous robots that react to light and to each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._light_sensitive_robot import LightSensitiveRobotElement
from ._swarm_robot import SwarmRobotElement

BEACON_MARGIN = 5.0


@register_system(
    field='engineering/robotics',
    tags=('robotics', 'active matter', 'delay', 'sde'),
    references=('Mijalkov et al. 2016', 'Argun et al. 2021'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def light_seeking_robots(
    n_robots: int = 20,
    *,
    delay: float = 5.0,
    V0: float = 1.0,
    V_inf: float = 0.1,
    I0: float = 1.0,
    r0: float = 1.0,
    tau: float = 1.0,
    R: float = 2.0,
) -> System:
    """Return light-sensitive robots in a circular well, whose sensory delay decides whether they gather in the light.

    Each robot moves at `V_inf + (V0 - V_inf) exp(-I(t - delay))` in a direction that diffuses with time scale `tau`,
    in the Gaussian light `I = I0 exp(-(x^2 + y^2) / r0^2)` at the centre of a well of radius `R`. The robots start
    at the centre, in directions spread around the circle, and do not interact. A positive delay drives them out of
    the light: a robot crosses the light at the fast speed it measured in the dark and lingers outside at the slow
    speed it measured in the light, so robots with a delay spend less time in the light than robots without one.

    Args:
        n_robots:
            The number of robots.

        delay:
            The sensory delay.

        V0:
            The speed in the dark.

        V_inf:
            The speed in strong light.

        I0:
            The intensity at the centre of the light.

        r0:
            The width of the light.

        tau:
            The time scale of rotational diffusion.

        R:
            The radius of the well.

    Returns:
        system:
            The system, with the group `robots` of positions `x`, `y` and directions `phi`.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import light_seeking_robots
        >>> light_seeking_robots(4).elements[0].prepare().size_
        4
    """
    headings = np.linspace(0.0, 2 * np.pi, n_robots, endpoint=False)
    robots = GroupElement(
        'robots',
        element=LightSensitiveRobotElement,
        size=n_robots,
        values={
            'x': np.zeros(n_robots),
            'y': np.zeros(n_robots),
            'phi': headings,
            'V0': V0,
            'V_inf': V_inf,
            'I0': I0,
            'r0': r0,
            'tau': tau,
            'R': R,
            'delay': delay,
        },
    )
    return System(elements=[robots])


@register_system(
    field='engineering/robotics',
    tags=('robotics', 'swarms', 'agents', 'taxis'),
    references=('Bjerknes, Winfield, and Melhuish 2007', 'Varughese et al. 2019'),
    defaults={'until': 300},
)
def swarmtaxis(
    n_robots: int = 20,
    *,
    size: float = 100.0,
    spread: float = 3.0,
    speed: float = 0.2,
    radio_range: float = 4.0,
    avoid_illuminated: float = 1.5,
    avoid_shadowed: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the swarmtaxis algorithm, which moves a swarm of robots towards a beacon that only some of them see.

    The robots start in a cluster near the left edge of a square arena, with the beacon at the middle of the right edge.
    Each robot keeps radio contact with the swarm by turning around when it loses a connection, and avoids its
    neighbours. Robots that see the beacon avoid at a larger radius than shadowed robots, which moves the centre of the
    swarm towards the beacon.

    Args:
        n_robots:
            The number of robots.

        size:
            The side of the arena.

        spread:
            The standard deviation of the initial positions around the start.

        speed:
            The distance a robot moves per step.

        radio_range:
            The range of the radio pings.

        avoid_illuminated:
            The avoidance radius of illuminated robots.

        avoid_shadowed:
            The avoidance radius of shadowed robots.

        random_state:
            The seed or generator of the initial positions and headings.

    Returns:
        system:
            The system, with the group `robots` of headings `heading` and illumination `illuminated`.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import swarmtaxis
        >>> swarmtaxis(5).elements[0].prepare().size_
        5
    """
    rng = np.random.default_rng(random_state)
    start = np.array([size / 5, size / 2])
    points = np.clip(start + rng.normal(0.0, spread, (n_robots, 2)), 0.0, size)
    robots = GroupElement(
        'robots',
        element=SwarmRobotElement,
        size=n_robots,
        values={
            'heading': rng.uniform(0.0, 2 * np.pi, n_robots),
            'speed': speed,
            'radio_range': radio_range,
            'alpha': float(n_robots),
            'avoid_illuminated': avoid_illuminated,
            'avoid_shadowed': avoid_shadowed,
            'beacon_x': size - BEACON_MARGIN,
            'beacon_y': size / 2,
            'arena': size,
        },
        location=[tuple(map(float, point)) for point in points],
    )
    return System(elements=[robots], space=ContinuousSpace(extent=(size, size)))
