"""Robot of the swarmtaxis algorithm, which keeps in radio contact with the swarm and avoids its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import AgentView, ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._swarm_robot_view import SwarmRobotView

EDGE_FRACTION = 1e-9


class SwarmRobotElement(Element):
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.engineering.robotics import SwarmRobotElement
        >>> GroupElement('robots', element=SwarmRobotElement, size=2, location=[(1.0, 1.0), (2.0, 1.0)]).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'illuminated', 'color_range': (0.0, 1.0)}),
        *,
        heading: Annotated[float, StateSpec()] = 0.0,
        connections: Annotated[float, StateSpec(domain=NN)] = 0.0,
        illuminated: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        radio_range: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        avoid_illuminated: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        avoid_shadowed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        shadow_angle: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beacon_x: Annotated[float, ParameterSpec()] = 95.0,
        beacon_y: Annotated[float, ParameterSpec()] = 50.0,
        arena: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
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

    @classmethod
    def _get_view(cls) -> type[SwarmRobotView]:
        """Return the view of the robots, which types their states and parameters."""
        return SwarmRobotView

    def _is_shadowed(self, agent: SwarmRobotView, neighbors: list[AgentView]) -> bool:
        """Return whether a neighbour lies in the cone from the robot towards the beacon."""
        x, y = SwarmRobotElement._position(agent)
        bearing = SwarmRobotElement._bearing(agent.beacon_x - x, agent.beacon_y - y)
        return any(
            SwarmRobotElement._angle_between(
                SwarmRobotElement._bearing(
                    SwarmRobotElement._position(other)[0] - x,
                    SwarmRobotElement._position(other)[1] - y,
                ),
                bearing,
            )
            < agent.shadow_angle
            for other in neighbors
        )

    def _turn(self, agent: SwarmRobotView, model: ModelView, count: int) -> float:
        """Return the new heading: back to the swarm after a lost connection, or at random after a new one."""
        if count < agent.connections and count < agent.alpha:
            return agent.heading + math.pi
        if count > agent.connections:
            return float(model.rng.uniform(0.0, 2 * math.pi))
        return agent.heading

    def _avoid(self, agent: SwarmRobotView, neighbors: list[AgentView], heading: float) -> float:
        """Return the heading away from the nearest neighbour inside the avoidance radius, or the heading unchanged."""
        x, y = SwarmRobotElement._position(agent)
        radius = agent.avoid_illuminated if agent.illuminated else agent.avoid_shadowed
        distances = [
            math.hypot(SwarmRobotElement._position(other)[0] - x, SwarmRobotElement._position(other)[1] - y)
            for other in neighbors
        ]
        close = [(distance, other) for distance, other in zip(distances, neighbors, strict=True) if distance < radius]
        if not close:
            return heading
        _, nearest = min(close, key=lambda pair: pair[0])
        return SwarmRobotElement._bearing(
            x - SwarmRobotElement._position(nearest)[0],
            y - SwarmRobotElement._position(nearest)[1],
        )

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the swarmtaxis rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.move)]

    def move(self, agent: SwarmRobotView, model: ModelView) -> None:
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
        x, y = SwarmRobotElement._position(agent)
        edge = agent.arena * (1 - EDGE_FRACTION)
        x = min(max(x + agent.speed * math.cos(heading), 0.0), edge)
        y = min(max(y + agent.speed * math.sin(heading), 0.0), edge)
        agent.move_to((x, y))

    @staticmethod
    def _position(agent: AgentView) -> tuple[float, float]:
        """Return the position of an agent in a continuous space."""
        x, y = np.asarray(agent.location, dtype=float)
        return float(x), float(y)

    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        """Return the angle of a vector."""
        return math.atan2(dy, dx)

    @staticmethod
    def _angle_between(first: float, second: float) -> float:
        """Return the absolute difference of two angles, between 0 and pi."""
        return abs((first - second + math.pi) % (2 * math.pi) - math.pi)
