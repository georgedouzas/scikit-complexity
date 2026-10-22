"""Robot models: autonomous robots that react to light and to each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import cos, exp, sin, sqrt

from ....core.base import Domain, Expression, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Boundary, Law, Noise, Rate, Rule, Symbols, delayed, where
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.space import ContinuousSpace
from ....core.system import System

BEACON_MARGIN = 5.0
EDGE_FRACTION = 1e-9


def _light(s: Symbols, x: Expression, y: Expression) -> Expression:
    """Return the Gaussian light intensity `I0 exp(-(x^2 + y^2) / r0^2)` at a point."""
    return s.I0 * exp(-(x**2 + y**2) / s.r0**2)


def _position(agent: AgentView) -> tuple[float, float]:
    """Return the position of an agent in a continuous space."""
    x, y = np.asarray(agent.location, dtype=float)
    return float(x), float(y)


def _bearing(dx: float, dy: float) -> float:
    """Return the angle of a vector."""
    return math.atan2(dy, dx)


def _angle_between(first: float, second: float) -> float:
    """Return the absolute difference of two angles, between 0 and pi."""
    return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


class LightSensitiveRobot(Element):
    """Robot in a Gaussian light field that slows down in light, reacting to the light it measured a delay ago.

    The robot moves at the speed `v = V_inf + (V0 - V_inf) exp(-I(t - delay))` in the direction `phi`, which diffuses
    with time scale `tau`, `dphi = sqrt(2 / tau) dW`. The light is `I(x, y) = I0 exp(-(x^2 + y^2) / r0^2)`, and a wall
    at radius `R` reflects the robot back into a circular well.

    Args:
        label:
            The label of the robot or the group.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        phi:
            The initial direction of motion.

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

        delay:
            The sensory delay, 0 for a robot that reacts to the light it measures now.

    Examples:
        >>> from skcomplex.models.engineering.robotics import LightSensitiveRobot
        >>> LightSensitiveRobot('robot', delay=5.0).delay
        5.0
    """

    x: float = state(default=0.0)
    y: float = state(default=0.0)
    phi: float = state(default=0.0)
    V0: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    V_inf: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    I0: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    r0: float = parameter(default=1.0, domain=Domain.POSITIVE)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)
    R: float = parameter(default=2.0, domain=Domain.POSITIVE)
    delay: float = 0.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the motion at the light-dependent speed, the rotational noise, and the reflecting wall.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        sensed = _light(s, delayed(s.x, self.delay), delayed(s.y, self.delay)) if self.delay else _light(s, s.x, s.y)
        speed = s.V_inf + (s.V0 - s.V_inf) * exp(-sensed)
        reach = where(sqrt(s.x**2 + s.y**2) > s.R, sqrt(s.x**2 + s.y**2), s.R)
        return [
            Rate(s.x, speed * cos(s.phi)),
            Rate(s.y, speed * sin(s.phi)),
            Noise(s.phi, sqrt(2 / s.tau)),
            Boundary(s.x, s.x * s.R / reach),
            Boundary(s.y, s.y * s.R / reach),
        ]


class SwarmRobot(Element):
    """Robot of the swarmtaxis algorithm, which keeps in radio contact with the swarm and avoids its neighbours.

    Every step the robot counts the robots within its radio range. When it loses a connection and has fewer than
    `alpha` connections, it turns around to stay with the swarm, and when it gains one it turns at random. A robot
    closer than its avoidance radius to another turns away from it. A robot is shadowed when another robot lies between
    it and the beacon, within the radio range and `shadow_angle` of the bearing to the beacon, and it is illuminated
    otherwise. Illuminated robots avoid at a larger radius than shadowed ones, so they move away from the swarm towards
    the beacon, and the swarm follows. The robot then moves forward at its speed.

    Args:
        label:
            The label of the agent or group.

        heading:
            The direction of motion.

        connections:
            The number of robots within radio range at the last step.

        illuminated:
            Whether the robot sees the beacon.

        speed:
            The distance moved per step.

        radio_range:
            The range of the radio pings.

        alpha:
            The number of connections below which a robot that loses one turns around.

        avoid_illuminated:
            The avoidance radius of illuminated robots.

        avoid_shadowed:
            The avoidance radius of shadowed robots.

        shadow_angle:
            The half angle of the cone towards the beacon in which another robot casts a shadow.

        beacon_x:
            The horizontal position of the beacon.

        beacon_y:
            The vertical position of the beacon.

        arena:
            The side of the square arena, whose walls stop the robot.

    Examples:
        >>> from skcomplex.models.engineering.robotics import SwarmRobot
        >>> SwarmRobot.group('robots', 2, location=[(1.0, 1.0), (2.0, 1.0)]).size
        2
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'illuminated', 'color_range': (0.0, 1.0)})
    heading: float = state(default=0.0)
    connections: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    illuminated: float = state(default=1.0, domain=Domain.BOOLEAN)
    speed: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    radio_range: float = parameter(default=4.0, domain=Domain.POSITIVE)
    alpha: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    avoid_illuminated: float = parameter(default=1.5, domain=Domain.NONNEGATIVE)
    avoid_shadowed: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    shadow_angle: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    beacon_x: float = parameter(default=95.0)
    beacon_y: float = parameter(default=50.0)
    arena: float = parameter(default=100.0, domain=Domain.POSITIVE)

    def _is_shadowed(self, agent: AgentView, neighbors: list[AgentView]) -> bool:
        """Return whether a neighbour lies in the cone from the robot towards the beacon."""
        x, y = _position(agent)
        bearing = _bearing(agent.beacon_x - x, agent.beacon_y - y)
        return any(
            _angle_between(_bearing(_position(other)[0] - x, _position(other)[1] - y), bearing) < agent.shadow_angle
            for other in neighbors
        )

    def _turn(self, agent: AgentView, model: ModelView, count: int) -> float:
        """Return the new heading: back to the swarm after a lost connection, or at random after a new one."""
        if count < agent.connections and count < agent.alpha:
            return agent.heading + math.pi
        if count > agent.connections:
            return float(model.rng.uniform(0.0, 2 * math.pi))
        return agent.heading

    def _avoid(self, agent: AgentView, neighbors: list[AgentView], heading: float) -> float:
        """Return the heading away from the nearest neighbour inside the avoidance radius, or the heading unchanged."""
        x, y = _position(agent)
        radius = agent.avoid_illuminated if agent.illuminated else agent.avoid_shadowed
        distances = [math.hypot(_position(other)[0] - x, _position(other)[1] - y) for other in neighbors]
        close = [(distance, other) for distance, other in zip(distances, neighbors, strict=True) if distance < radius]
        if not close:
            return heading
        _, nearest = min(close, key=lambda pair: pair[0])
        return _bearing(x - _position(nearest)[0], y - _position(nearest)[1])

    def laws(self, s: Symbols) -> list[Law]:
        """Return the swarmtaxis rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.move)]

    def move(self, agent: AgentView, model: ModelView) -> None:
        """Update the connections, the shadow, and the heading, and move forward.

        Args:
            agent:
                The robot.

            model:
                The model.
        """
        neighbors = agent.neighbors(agent.radio_range)
        count = len(neighbors)
        agent.illuminated = 0.0 if self._is_shadowed(agent, neighbors) else 1.0
        heading = self._avoid(agent, neighbors, self._turn(agent, model, count))
        agent.connections = count
        agent.heading = heading % (2 * math.pi)
        x, y = _position(agent)
        edge = agent.arena * (1 - EDGE_FRACTION)
        x = min(max(x + agent.speed * math.cos(heading), 0.0), edge)
        y = min(max(y + agent.speed * math.sin(heading), 0.0), edge)
        agent.move_to((x, y))


@register_model(
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
            The model, with the group `robots` of positions `x`, `y` and directions `phi`.

    Examples:
        >>> from skcomplex.models.engineering.robotics import light_seeking_robots
        >>> light_seeking_robots(4).elements[0].size
        4
    """
    headings = np.linspace(0.0, 2 * np.pi, n_robots, endpoint=False)
    robots = LightSensitiveRobot.group(
        'robots',
        n_robots,
        x=np.zeros(n_robots),
        y=np.zeros(n_robots),
        phi=headings,
        V0=V0,
        V_inf=V_inf,
        I0=I0,
        r0=r0,
        tau=tau,
        R=R,
        delay=delay,
    )
    return System(components=[robots])


@register_model(
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
            The model, with the group `robots` of headings `heading` and illumination `illuminated`.

    Examples:
        >>> from skcomplex.models.engineering.robotics import swarmtaxis
        >>> swarmtaxis(5).elements[0].size
        5
    """
    rng = np.random.default_rng(random_state)
    start = np.array([size / 5, size / 2])
    points = np.clip(start + rng.normal(0.0, spread, (n_robots, 2)), 0.0, size)
    robots = SwarmRobot.group(
        'robots',
        n_robots,
        location=[tuple(map(float, point)) for point in points],
        heading=rng.uniform(0.0, 2 * np.pi, n_robots),
        speed=speed,
        radio_range=radio_range,
        alpha=float(n_robots),
        avoid_illuminated=avoid_illuminated,
        avoid_shadowed=avoid_shadowed,
        beacon_x=size - BEACON_MARGIN,
        beacon_y=size / 2,
        arena=size,
    )
    return System(components=[robots], space=ContinuousSpace(size, size))
