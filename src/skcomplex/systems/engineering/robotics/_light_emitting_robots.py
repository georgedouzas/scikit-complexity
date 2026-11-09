"""Light-emitting robots: robots that slow down in the light of the others.

The light is sensed with a delay or a prediction.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class LightEmittingRobotView(AgentView):
    """View of a light-emitting robot, with its states as attributes."""

    phi: float
    light: float
    speed: float


class LightEmittingRobotElement(Element):
    """Light-emitting robot that moves in the direction `phi`.

    Its speed is set by the light of the others it senses.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phi:
            The initial direction of motion.

        light:
            The light of the other robots the robot measures now.

        speed:
            The speed the robot moves at.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import LightEmittingRobotElement
        >>> LightEmittingRobotElement('robot', phi=1.0).phi
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'light'}),
        *,
        phi: Annotated[float, StateSpec(role='direction')] = 0.0,
        light: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        speed: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.light = light
        self.speed = speed

    @classmethod
    def _get_view(cls) -> type[LightEmittingRobotView]:
        """Return the view of the robots.

        It types their states.
        """
        return LightEmittingRobotView


class SensoryDelayInteraction(Interaction):
    """Motion of light-emitting robots whose speed follows the light of the others.

    The light is sensed with a delay. Every robot emits the Gaussian light `I0 exp(-d^2 / r0^2)` at the distance `d`,
    and measures the total light `I_i` of the other robots, at the shortest distances on the torus. It moves at the
    speed `V_inf + (V0 - V_inf) exp(-I_i(t - delay))` in its direction, which diffuses with the time scale `tau`,
    `phi <- phi + sqrt(2 step / tau) w` with a standard normal `w`. A positive delay reads the light measured that long
    ago, and a negative delay predicts it by the linear extrapolation `I(t) - delay I'(t)`, never below 0. Before the
    delay has passed, a robot uses the earliest light it measured. Every step of the rule lasts `step`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of robots.

        I0:
            The intensity of the light of every robot.

        r0:
            The width of the light of every robot.

        V0:
            The speed in the dark.

        V_inf:
            The speed in strong light.

        tau:
            The time scale of rotational diffusion.

        delay:
            The sensory delay, negative for a prediction.

        step:
            The time of one step.

        side:
            The side of the periodic square arena.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import SensoryDelayInteraction
        >>> SensoryDelayInteraction('motion', target='robots', delay=-10.0).delay
        -10.0
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        I0: float = 5.0,
        r0: float = 1.0,
        V0: float = 1.0,
        V_inf: float = 0.1,
        tau: float = 2.0,
        delay: float = 0.0,
        step: float = 0.5,
        side: float = 10.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.I0 = I0
        self.r0 = r0
        self.V0 = V0
        self.V_inf = V_inf
        self.tau = tau
        self.delay = delay
        self.step = step
        self.side = side

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the motion of the robots.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule of the motion.
        """
        return [RuleLaw(self._move)]

    def _move(self, system: SystemView) -> None:
        """Measure the light of the others, set the speeds, and move the robots.

        The speeds follow the delayed or predicted light.
        """
        robots = system.agents('target', LightEmittingRobotView)
        positions = np.array([np.asarray(robot.location, dtype=float) for robot in robots])
        light = self._light(positions)
        history = cast('list[Array]', system.storage.setdefault('light', []))
        history.append(light)
        sensed = self._sensed(history)
        speeds = self.V_inf + (self.V0 - self.V_inf) * np.exp(-sensed)
        spread = math.sqrt(2 * self.step / self.tau)
        for robot, position, measured, speed in zip(robots, positions, light, speeds, strict=True):
            step = speed * self.step * np.array([math.cos(robot.phi), math.sin(robot.phi)])
            robot.move_to(tuple(np.mod(position + step, self.side)))
            robot.phi = float(robot.phi + spread * system.rng.normal())
            robot.light = float(measured)
            robot.speed = float(speed)

    def _light(self, positions: Array) -> Array:
        """Return the total light every robot measures from the others.

        The light is taken at the shortest distances on the torus.
        """
        offsets = positions[:, None, :] - positions[None, :, :]
        offsets -= self.side * np.round(offsets / self.side)
        glow = self.I0 * np.exp(-(offsets**2).sum(axis=-1) / self.r0**2)
        np.fill_diagonal(glow, 0.0)
        light: Array = glow.sum(axis=1)
        return light

    def _sensed(self, history: list[Array]) -> Array:
        """Return the light the robots react to.

        It is measured a delay ago, or predicted ahead for a negative delay.
        """
        lag = round(abs(self.delay) / self.step)
        if self.delay >= 0:
            sensed: Array = history[max(len(history) - 1 - lag, 0)]
            return sensed
        slope = (history[-1] - history[-2]) / self.step if len(history) > 1 else np.zeros_like(history[-1])
        predicted: Array = np.maximum(history[-1] - self.delay * slope, 0.0)
        return predicted


class LightEmittingRobotsSystem(BaseSystem):
    """Light-emitting robots that cluster or spread by their sensing of light.

    They cluster when they sense the light of the others late and spread when early. Every robot emits a Gaussian light
    and measures the total light of the others, `I_i = sum_j I0 exp(-d_ij^2 / r0^2)` over the shortest distances on a
    periodic square arena. It moves at the speed `V_inf + (V0 - V_inf) exp(-I_i(t - delay))` in a direction that
    diffuses with the time scale `tau`. A robot with a positive delay crosses a dark region at the fast speed it sensed
    earlier in the dark and slows down only once it is in the light, so the robots linger near each other and form
    clusters. A negative delay, a prediction of the light by its trend, makes a robot slow down before it reaches the
    light and speed up before it leaves it, so the robots avoid each other and spread over the arena. Every step lasts
    `step`, so the default run of 400 steps covers 200 s.

    The system has the group `robots` of directions `phi`, measured lights `light`, and speeds `speed`, at positions in
    the arena.

    Args:
        n_robots:
            The number of robots.

        delay:
            The sensory delay, negative for a prediction.

        side:
            The side of the periodic square arena.

        I0:
            The intensity of the light of every robot.

        r0:
            The width of the light of every robot.

        V0:
            The speed in the dark.

        V_inf:
            The speed in strong light.

        tau:
            The time scale of rotational diffusion.

        step:
            The time of one step.

        random_state:
            The seed or generator of the initial positions and directions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.robotics import LightEmittingRobotsSystem
        >>> LightEmittingRobotsSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'engineering/robotics'
    topics = ('robotics', 'active matter', 'delay', 'agents', 'swarms')
    references = ('Mijalkov et al. 2016', 'Argun et al. 2021')
    until = 400

    def __init__(
        self,
        n_robots: int = 50,
        *,
        delay: float = 10.0,
        side: float = 10.0,
        I0: float = 5.0,
        r0: float = 1.0,
        V0: float = 1.0,
        V_inf: float = 0.1,
        tau: float = 2.0,
        step: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_robots = n_robots
        self.delay = delay
        self.side = side
        self.I0 = I0
        self.r0 = r0
        self.V0 = V0
        self.V_inf = V_inf
        self.tau = tau
        self.step = step
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        positions = rng.uniform(0.0, self.side, (self.n_robots, 2))
        robots = GroupElement(
            'robots',
            element=LightEmittingRobotElement,
            size=self.n_robots,
            values={'phi': rng.uniform(0.0, 2 * np.pi, self.n_robots), 'speed': self.V0},
            location=[tuple(map(float, point)) for point in positions],
        )
        motion = SensoryDelayInteraction(
            'motion',
            target='robots',
            I0=self.I0,
            r0=self.r0,
            V0=self.V0,
            V_inf=self.V_inf,
            tau=self.tau,
            delay=self.delay,
            step=self.step,
            side=self.side,
        )
        return System(
            self.label,
            elements=[robots],
            interactions=[motion],
            space=ContinuousSpace(extent=(self.side, self.side), boundary='periodic'),
        )
