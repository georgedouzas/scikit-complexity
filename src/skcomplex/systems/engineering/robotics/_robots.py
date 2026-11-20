"""Robot systems: autonomous robots.

They react to light and to each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, cos, exp, oo, sin, sqrt
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Period, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role

EDGE_FRACTION = 1e-9
BEACON_MARGIN = 5.0


class LightSensitiveRobotElement(Element):
    """Robot at the position `x`, `y` that moves in the direction `phi`.

    `LightSensitivityInteraction` sets its speed from the light it measures.

    Args:
        label:
            The label of the robot or the group.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        phi:
            The initial direction of motion.

        size:
            The number of robots, or `None` for one.

        space:
            The plane, a continuous space whose axes `x` and `y` make the position of a robot its place, or `None`.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import LightSensitiveRobotElement
        >>> LightSensitiveRobotElement('robot', phi=1.0).phi
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        y: object = 0.0,
        phi: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.y = y
        self.phi = phi


class LightSensitivityInteraction(Interaction):
    """Motion of a robot in a Gaussian light field that slows down in light.

    It reacts to the light it measured a delay ago. The robot moves at the speed
    `v = V_inf + (V0 - V_inf) exp(-I(t - delay))` in the direction `phi`, which diffuses with time scale `tau`,
    `dphi = sqrt(2 / tau) dW`. The light is `I(x, y) = I0 exp(-(x^2 + y^2) / r0^2)`, and a wall at radius `R` reflects
    the robot back into a circular well.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the robot or the group of robots.

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
        >>> from skcomplex.systems.engineering.robotics import LightSensitivityInteraction
        >>> LightSensitivityInteraction('light', target='robots', delay=5.0).delay
        5.0
    """

    law_classes = (RateLaw, NoiseLaw, UpdateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        V0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        V_inf: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        I0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        r0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
        R: Annotated[float, RealSet.open(0, oo)] = 2.0,
        delay: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.V0 = V0
        self.V_inf = V_inf
        self.I0 = I0
        self.r0 = r0
        self.tau = tau
        self.R = R
        self.delay = delay

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion at the light-dependent speed, the rotational noise, and the reflecting wall.

        Args:
            context:
                The namespace of the robots, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, t = context, context.t
        x, y, phi = s.target.x, s.target.y, s.target.phi
        sensed = (
            LightSensitivityInteraction._get_light(s, x(t=t - self.delay), y(t=t - self.delay))
            if self.delay
            else LightSensitivityInteraction._get_light(s, x, y)
        )
        speed = s.V_inf + (s.V0 - s.V_inf) * exp(-sensed)
        reach = where(sqrt(x**2 + y**2) > s.R, sqrt(x**2 + y**2), s.R)
        self.laws_: list[BaseComponent] = [
            RateLaw(x.diff(t), speed * cos(phi)),
            RateLaw(y.diff(t), speed * sin(phi)),
            NoiseLaw(phi, sqrt(2 / s.tau)),
            UpdateLaw(x, x * s.R / reach),
            UpdateLaw(y, y * s.R / reach),
        ]

    @staticmethod
    def _get_light(s: SimpleNamespace, x: Expression, y: Expression) -> Expression:
        """Return the Gaussian light intensity `I0 exp(-(x^2 + y^2) / r0^2)` at a point."""
        return s.I0 * exp(-(x**2 + y**2) / s.r0**2)


class SwarmRobotView(AgentViewProtocol, Protocol):
    """View of a robot of the swarmtaxis algorithm.

    Its states and parameters are attributes.
    """

    heading: float
    connections: float
    illuminated: float
    speed: float
    radio_range: float
    alpha: float
    avoid_illuminated: float
    avoid_shadowed: float
    shadow_angle: float
    beacon_x: float
    beacon_y: float
    arena: float


class SwarmRobotElement(Element):
    """Robot of the swarmtaxis algorithm.

    It keeps in radio contact with the swarm and avoids its neighbours, by the rule of `SwarmtaxisInteraction`.

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

        values:
            Further values by name, such as the position `x` and `y` of every robot.

        size:
            The number of robots, or `None` for one.

        space:
            The arena, a continuous space whose axes `x` and `y` place the robots, or `None` for none.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import SwarmRobotElement
        >>> SwarmRobotElement('robots', values={'x': [1.0, 2.0], 'y': 1.0}, size=2).size
        2
    """

    visual = VisualSpec('agents', {'color_by': 'illuminated', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        heading: Annotated[object, Period(2 * math.pi)] = 0.0,
        connections: Annotated[float, NN] = 0.0,
        illuminated: Annotated[float, FiniteEnumeratedSet([0, 1])] = 1.0,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
        radio_range: Annotated[float, RealSet.open(0, oo)] = 4.0,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 20.0,
        avoid_illuminated: Annotated[float, RealSet.closed_open(0, oo)] = 1.5,
        avoid_shadowed: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        shadow_angle: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        beacon_x: float = 95.0,
        beacon_y: float = 50.0,
        arena: Annotated[float, RealSet.open(0, oo)] = 100.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.heading = heading
        self.connections = connections
        self.illuminated = illuminated
        self.speed = speed
        self.radio_range = radio_range
        self.alpha = alpha
        self.avoid_illuminated = avoid_illuminated
        self.avoid_shadowed = avoid_shadowed
        self.shadow_angle = shadow_angle
        self.beacon_x = beacon_x
        self.beacon_y = beacon_y
        self.arena = arena


class SwarmtaxisInteraction(Interaction):
    """Rule of the swarmtaxis algorithm, which moves a swarm of robots towards a beacon.

    Every step a robot counts the robots within its radio range. When it loses a connection and has fewer than `alpha`
    connections, it turns around to stay with the swarm, and when it gains one it turns at random. A robot closer than
    its avoidance radius to another turns away from it. A robot is shadowed when another robot lies between it and the
    beacon, within the radio range and `shadow_angle` of the bearing to the beacon, and it is illuminated otherwise.
    Illuminated robots avoid at a larger radius than shadowed ones, so they move away from the swarm towards the beacon,
    and the swarm follows. The robot then moves forward at its speed.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of robots.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import SwarmtaxisInteraction
        >>> SwarmtaxisInteraction('swarmtaxis', target='robots').target
        'robots'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the swarmtaxis rule of every robot."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.move)]

    def _is_shadowed(self, agent: SwarmRobotView, neighbors: Sequence[AgentViewProtocol]) -> bool:
        """Return whether a neighbour lies in the cone from the robot towards the beacon."""
        x, y = SwarmtaxisInteraction._get_position(agent)
        bearing = SwarmtaxisInteraction._get_bearing(agent.beacon_x - x, agent.beacon_y - y)
        return any(
            SwarmtaxisInteraction._get_angle_between(
                SwarmtaxisInteraction._get_bearing(
                    SwarmtaxisInteraction._get_position(other)[0] - x,
                    SwarmtaxisInteraction._get_position(other)[1] - y,
                ),
                bearing,
            )
            < agent.shadow_angle
            for other in neighbors
        )

    def _turn(self, agent: SwarmRobotView, system: SystemView, count: int) -> float:
        """Return the new heading.

        It turns back to the swarm after a lost connection, or at random after a new one.
        """
        if count < agent.connections and count < agent.alpha:
            return agent.heading + math.pi
        if count > agent.connections:
            return float(system.rng.uniform(0.0, 2 * math.pi))
        return agent.heading

    def _avoid(self, agent: SwarmRobotView, neighbors: Sequence[AgentViewProtocol], heading: float) -> float:
        """Return the heading away from the nearest neighbour inside the avoidance radius.

        Without such a neighbour the heading is unchanged.
        """
        x, y = SwarmtaxisInteraction._get_position(agent)
        radius = agent.avoid_illuminated if agent.illuminated else agent.avoid_shadowed
        distances = [
            math.hypot(
                SwarmtaxisInteraction._get_position(other)[0] - x,
                SwarmtaxisInteraction._get_position(other)[1] - y,
            )
            for other in neighbors
        ]
        close = [(distance, other) for distance, other in zip(distances, neighbors, strict=True) if distance < radius]
        if not close:
            return heading
        _, nearest = min(close, key=lambda pair: pair[0])
        return SwarmtaxisInteraction._get_bearing(
            x - SwarmtaxisInteraction._get_position(nearest)[0],
            y - SwarmtaxisInteraction._get_position(nearest)[1],
        )

    def move(self, agent: SwarmRobotView, system: SystemView) -> None:
        """Update the connections, the shadow, and the heading, and move forward.

        Args:
            agent:
                The robot.

            system:
                The system.
        """
        neighbors = agent.get_neighbors(agent.radio_range)
        count = len(neighbors)
        agent.illuminated = 0.0 if self._is_shadowed(agent, neighbors) else 1.0
        heading = self._avoid(agent, neighbors, self._turn(agent, system, count))
        agent.connections = count
        agent.heading = heading % (2 * math.pi)
        x, y = SwarmtaxisInteraction._get_position(agent)
        edge = agent.arena * (1 - EDGE_FRACTION)
        x = min(max(x + agent.speed * math.cos(heading), 0.0), edge)
        y = min(max(y + agent.speed * math.sin(heading), 0.0), edge)
        agent.move_to((x, y))

    @staticmethod
    def _get_position(agent: AgentViewProtocol) -> tuple[float, float]:
        """Return the position of an agent in a continuous space."""
        x, y = np.asarray(agent.location, dtype=float)
        return float(x), float(y)

    @staticmethod
    def _get_bearing(dx: float, dy: float) -> float:
        """Return the angle of a vector."""
        return math.atan2(dy, dx)

    @staticmethod
    def _get_angle_between(first: float, second: float) -> float:
        """Return the absolute difference of two angles, between 0 and pi."""
        return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


class LightSeekingRobotsSystem(BaseSystem):
    """Light-sensitive robots in a circular well.

    Their sensory delay decides whether they gather in the light. Each robot moves at
    `V_inf + (V0 - V_inf) exp(-I(t - delay))` in a direction that diffuses with time scale `tau`, in the Gaussian light
    `I = I0 exp(-(x^2 + y^2) / r0^2)` at the centre of a well of radius `R`. The robots start at the centre, in
    directions spread around the circle, and do not interact. A positive delay drives them out of the light: a robot
    crosses the light at the fast speed it measured in the dark and lingers outside at the slow speed it measured in the
    light, so robots with a delay spend less time in the light than robots without one.

    The system has the group `robots` of positions `x`, `y` and directions `phi`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import LightSeekingRobotsSystem
        >>> LightSeekingRobotsSystem(4).prepare().system_.elements[0].size
        4
    """

    field = 'engineering/robotics'
    topics = ('robotics', 'active matter', 'delay', 'sde')
    references = ('Mijalkov et al. 2016', 'Argun et al. 2021')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        n_robots: int = 20,
        *,
        delay: float = 5.0,
        V0: float = 1.0,
        V_inf: float = 0.1,
        I0: float = 1.0,
        r0: float = 1.0,
        tau: float = 1.0,
        R: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_robots = n_robots
        self.delay = delay
        self.V0 = V0
        self.V_inf = V_inf
        self.I0 = I0
        self.r0 = r0
        self.tau = tau
        self.R = R

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        headings = np.linspace(0.0, 2 * np.pi, self.n_robots, endpoint=False)
        robots = LightSensitiveRobotElement(
            'robots',
            x=np.zeros(self.n_robots),
            y=np.zeros(self.n_robots),
            phi=headings,
            size=self.n_robots,
            space=ContinuousSpace(n_dim=2),
        )
        light = LightSensitivityInteraction(
            'light',
            target='robots',
            V0=self.V0,
            V_inf=self.V_inf,
            I0=self.I0,
            r0=self.r0,
            tau=self.tau,
            R=self.R,
            delay=self.delay,
        )
        self.system_ = System(self.label, elements=[robots], interactions=[light])


class SwarmtaxisSystem(BaseSystem):
    """The swarmtaxis algorithm, which moves a swarm of robots towards a beacon.

    Only some of the robots see the beacon. The robots start in a cluster near the left edge of a square arena, with the
    beacon at the middle of the right edge. Each robot keeps radio contact with the swarm by turning around when it
    loses a connection, and avoids its neighbours. Robots that see the beacon avoid at a larger radius than shadowed
    robots, which moves the centre of the swarm towards the beacon. The system has the group `robots` of headings
    `heading` and illumination `illuminated`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import SwarmtaxisSystem
        >>> SwarmtaxisSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'engineering/robotics'
    topics = ('robotics', 'swarms', 'agents', 'taxis')
    references = ('Bjerknes, Winfield, and Melhuish 2007', 'Varughese et al. 2019')
    until = 300

    def __init__(
        self,
        n_robots: int = 20,
        *,
        size: float = 100.0,
        spread: float = 3.0,
        speed: float = 0.2,
        radio_range: float = 4.0,
        avoid_illuminated: float = 1.5,
        avoid_shadowed: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_robots = n_robots
        self.size = size
        self.spread = spread
        self.speed = speed
        self.radio_range = radio_range
        self.avoid_illuminated = avoid_illuminated
        self.avoid_shadowed = avoid_shadowed
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        start = np.array([self.size / 5, self.size / 2])
        points = np.clip(start + rng.normal(0.0, self.spread, (self.n_robots, 2)), 0.0, self.size)
        arena = ContinuousSpace(extent=(self.size, self.size))
        robots = SwarmRobotElement(
            'robots',
            heading=rng.uniform(0.0, 2 * np.pi, self.n_robots),
            speed=self.speed,
            radio_range=self.radio_range,
            alpha=float(self.n_robots),
            avoid_illuminated=self.avoid_illuminated,
            avoid_shadowed=self.avoid_shadowed,
            beacon_x=self.size - BEACON_MARGIN,
            beacon_y=self.size / 2,
            arena=self.size,
            values={'x': points[:, 0], 'y': points[:, 1]},
            size=self.n_robots,
            space=arena,
        )
        self.system_ = System(
            self.label,
            elements=[robots],
            interactions=[SwarmtaxisInteraction('swarmtaxis', target='robots')],
            space=arena,
        )
