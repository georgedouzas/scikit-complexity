"""Mechanics systems.

The module covers pendulums, oscillators, projectiles, and collisions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, GradientInteraction, Interaction, RateInteraction
from ....core.components.laws import BaseLaw, CrossingLaw, RateLaw
from ....core.components.laws.utils import derivative
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.mechanics import PendulumElement, PendulumInteraction, ReflectionInteraction
from ..common.particles import (
    ConstantForceInteraction,
    DampingInteraction,
    HarmonicForceInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
)

WALL_POINTS = 200
GRAVITY = 9.81


class BouncingBallElement(Element):
    """Ball at the height `h` above a floor.

    The ball falls under the gravitational acceleration `g`.

    The height has order 2 and its first derivative `h_d1` is the vertical velocity.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

        h:
            The initial height.

        h_d1:
            The initial vertical velocity.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BouncingBallElement
        >>> BouncingBallElement('ball', h=2.0).h
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(order=2)] = 1.0,
        h_d1: float = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.h_d1 = h_d1
        self.g = g


def _get_disc_wall(radius: float, **_: float) -> list[np.ndarray]:
    """Return the wall of a disc table as one closed polyline."""
    angles = np.linspace(0.0, 2 * np.pi, WALL_POINTS)
    return [radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class DiscBilliardElement(Element):
    """Particle moving freely in the plane inside a disc.

    The particle has the position `(x, y)` and the velocity `(vx, vy)`.

    The position changes at the velocity, which changes only at the reflections off the wall of the disc.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for its class visual.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

        radius:
            The radius of the disc.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DiscBilliardElement
        >>> DiscBilliardElement('particle', radius=2.0).radius
        2.0
    """

    law_classes = (RateLaw,)

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _get_disc_wall})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.3,
        y: Annotated[float, StateSpec()] = 0.0,
        vx: Annotated[float, StateSpec()] = 0.4,
        vy: Annotated[float, StateSpec()] = 1.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the position.
        """
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]


class HangingBallElement(Element):
    """Ball hanging on a string of length `length`.

    The ball swings with small amplitude about its rest position.

    The horizontal position `x` has order 2, and its first derivative `x_d1` is the velocity. For small amplitudes
    the ball moves in the potential `g (x - rest)² / (2 length)`.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

        x:
            The initial horizontal position.

        x_d1:
            The initial horizontal velocity.

        rest:
            The rest position.

        g:
            The gravitational acceleration.

        length:
            The length of the string.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import HangingBallElement
        >>> HangingBallElement('ball', rest=0.5).rest
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        rest: Annotated[float, ParameterSpec()] = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        length: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.rest = rest
        self.g = g
        self.length = length


class ElasticCollisionInteraction(Interaction):
    """Elastic collision of two equal masses in a row.

    The masses exchange their velocities when they touch.

    The collision is the event at which the gap `x_right - x_left - diameter` between the balls falls to zero, and
    equal masses swap their velocities in an elastic collision.

    Args:
        label:
            The label of the interaction.

        left:
            The path of the left ball, with a position `x` of order 2.

        right:
            The path of the right ball, with a position `x` of order 2.

        diameter:
            The diameter of the balls.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ElasticCollisionInteraction
        >>> ElasticCollisionInteraction('collision', left='a', right='b', diameter=0.2).diameter
        0.2
    """

    law_classes = (CrossingLaw,)

    def __init__(
        self,
        label: str,
        *,
        left: Annotated[str, RoleSpec()],
        right: Annotated[str, RoleSpec()],
        diameter: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.diameter = diameter

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the crossing law of the contact.

        The law swaps the velocities of the balls.

        Args:
            s:
                The namespace of the symbols the laws can use, with `s.left` and `s.right`.

        Returns:
            laws:
                The crossing law.
        """
        return [
            CrossingLaw(
                s.right.x - s.left.x - s.diameter,
                {
                    derivative(s.left.x, 1): derivative(s.right.x, 1),
                    derivative(s.right.x, 1): derivative(s.left.x, 1),
                },
                direction=-1,
            ),
        ]


class ProjectileSystem(BaseSystem):
    """A particle launched under uniform gravity, optionally slowed by linear drag.

    The system has the particle `ball` in two dimensions.

    Args:
        speed:
            The launch speed.

        angle:
            The launch angle above the horizon, in degrees.

        mass:
            The mass.

        drag:
            The linear drag coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ProjectileSystem
        >>> ProjectileSystem(speed=5.0, angle=90.0).prepare().system_.elements[0].x_d1[1]
        5.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode')
    references = ('Newton 1687',)
    until = 2.0
    dt = 0.01

    def __init__(
        self,
        speed: float = 10.0,
        angle: float = 45.0,
        *,
        mass: float = 1.0,
        drag: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.speed = speed
        self.angle = angle
        self.mass = mass
        self.drag = drag

    def _build_system(self) -> CompositeSystem:
        """Return a particle launched under uniform gravity.

        Linear drag optionally slows the particle.
        """
        radians = np.radians(self.angle)
        velocity = [round(float(self.speed * np.cos(radians)), 12), round(float(self.speed * np.sin(radians)), 12)]
        interactions: list[BaseInteraction] = [
            ConstantForceInteraction('gravity', target='ball', F=[0.0, -GRAVITY * self.mass]),
        ]
        if self.drag:
            interactions.append(DampingInteraction('drag', target='ball', gamma=self.drag))
        return CompositeSystem(
            self.label,
            elements=[MassPointElement('ball', x=[0.0, 0.0], x_d1=velocity, m=self.mass)],
            interactions=interactions,
            space=ContinuousSpace(n_dim=2),
        )


class HarmonicOscillatorSystem(BaseSystem):
    """A particle on a spring, optionally damped.

    The system has the particle `mass` in one dimension.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        k:
            The stiffness.

        mass:
            The mass.

        damping:
            The linear damping coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem
        >>> HarmonicOscillatorSystem(k=4.0).prepare().system_.interactions[0].k
        4.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'oscillation')
    references = ('Hooke 1678',)
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.0,
        *,
        k: float = 1.0,
        mass: float = 1.0,
        damping: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.k = k
        self.mass = mass
        self.damping = damping

    def _build_system(self) -> CompositeSystem:
        """Return a particle on a spring, optionally damped."""
        interactions: list[BaseInteraction] = [HarmonicPotentialInteraction('spring', target='mass', k=self.k)]
        if self.damping:
            interactions.append(DampingInteraction('damping', target='mass', gamma=self.damping))
        return CompositeSystem(
            self.label,
            elements=[MassPointElement('mass', x=self.x, x_d1=self.v, m=self.mass)],
            interactions=interactions,
            space=ContinuousSpace(n_dim=1),
        )


class PendulumSystem(BaseSystem):
    """A pendulum, whose angle follows Newton's second law `m l θ'' = -m g sin θ`.

    The system has the pendulum `bob`, whose angle has order 2 and angular velocity `bob.theta_d1`.

    Args:
        theta:
            The initial angle, in radians.

        omega:
            The initial angular velocity.

        m:
            The mass of the bob.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import PendulumSystem
        >>> PendulumSystem(theta=0.1).prepare().system_.elements[0].theta_d1
        0.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'oscillation')
    references = ('Newton 1687',)
    until = 10.0
    dt = 0.01

    def __init__(
        self,
        theta: float = 0.5,
        omega: float = 0.0,
        *,
        m: float = 1.0,
        g: float = 9.81,
        l: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.omega = omega
        self.m = m
        self.g = g
        self.l = l

    def _build_system(self) -> CompositeSystem:
        """Return a pendulum.

        Its angle follows Newton's second law `m l θ'' = -m g sin θ`.
        """
        return CompositeSystem(
            self.label,
            elements=[PendulumElement('bob', theta=self.theta, theta_d1=self.omega, m=self.m, g=self.g, l=self.l)],
            interactions=[PendulumInteraction('swing', target='bob')],
        )


class DrivenDampedOscillatorSystem(BaseSystem):
    """A mass on a spring with damping and a periodic force.

    The mass follows `m x'' = -k x - c x' + F cos(ω t)`.

    The system has the mass `mass` in one dimension, whose displacement `mass.x` has order 2.

    Args:
        x:
            The initial displacement.

        v:
            The initial velocity.

        m:
            The mass.

        k:
            The stiffness.

        c:
            The damping coefficient.

        F:
            The amplitude of the driving force.

        omega:
            The angular frequency of the driving force.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DrivenDampedOscillatorSystem
        >>> DrivenDampedOscillatorSystem(F=0.5).prepare().system_.interactions[2].F
        0.5
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'oscillation', 'forcing')
    references = ('Hooke 1678',)
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.0,
        *,
        m: float = 1.0,
        k: float = 1.0,
        c: float = 0.1,
        F: float = 0.0,
        omega: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.F = F
        self.omega = omega

    def _build_system(self) -> CompositeSystem:
        """Return a mass on a spring with damping and a periodic force.

        The mass follows `m x'' = -k x - c x' + F cos(ω t)`.
        """
        return CompositeSystem(
            self.label,
            elements=[MassPointElement('mass', x=self.x, x_d1=self.v, m=self.m)],
            interactions=[
                HarmonicPotentialInteraction('spring', target='mass', k=self.k),
                DampingInteraction('damping', target='mass', gamma=self.c),
                HarmonicForceInteraction('drive', target='mass', F=self.F, omega=self.omega),
            ],
            space=ContinuousSpace(n_dim=1),
        )


class BouncingBallSystem(BaseSystem):
    """A ball that falls under gravity and bounces off the floor.

    The ball loses speed by a coefficient of restitution.

    Every impact is an event that reverses the velocity and scales it by `e`. The impacts accumulate at the Zeno time
    `sqrt(2 h / g) (1 + e) / (1 - e)` for a ball dropped from rest, so runs should end before it. The system has the
    height `ball.h` of order 2 and its velocity `ball.h_d1`.

    Args:
        h:
            The initial height.

        v:
            The initial vertical velocity.

        g:
            The gravitational acceleration.

        e:
            The coefficient of restitution.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BouncingBallSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> trajectory = simulate(BouncingBallSystem(), until=1.0, dt=0.01)
        >>> round(trajectory.events[0].time, 6)
        0.451524
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'events', 'impacts')
    references = ('Strogatz 2015', 'Hairer, Nørsett, and Wanner 1993')
    until = 3.0
    dt = 0.01

    def __init__(
        self,
        h: float = 1.0,
        v: float = 0.0,
        *,
        g: float = 9.81,
        e: float = 0.8,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.h = h
        self.v = v
        self.g = g
        self.e = e

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[BouncingBallElement('ball', h=self.h, h_d1=self.v, g=self.g)],
            interactions=[
                RateInteraction('gravity', target='ball', state='h', rate=lambda s: -s.target.g),
                ReflectionInteraction(
                    'impact',
                    target='ball',
                    guard=lambda s: s.target.h,
                    positions=('h',),
                    velocities=('h_d1',),
                    direction=-1,
                    restitution=self.e,
                ),
            ],
        )


class NewtonsCradleSystem(BaseSystem):
    """Newton's cradle.

    Pendulum balls in a row pass momentum along by elastic collisions.

    The balls swing with small amplitude about rest positions that leave a small gap between neighbours, so collisions
    are separate events. The first ball starts with a speed towards the others, and equal masses exchange velocities at
    every collision. The system has the horizontal positions `ball_<i>.x` and velocities `ball_<i>.x_d1`.

    Args:
        n_balls:
            The number of balls.

        speed:
            The initial speed of the first ball.

        diameter:
            The diameter of the balls.

        gap:
            The gap between neighbouring balls at rest.

        length:
            The length of the pendulums.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import NewtonsCradleSystem
        >>> len(NewtonsCradleSystem(n_balls=3).prepare().system_.interactions)
        5
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'events', 'impacts', 'momentum')
    references = ('Hinch and Saint-Jean 1999',)
    until = 5.0
    dt = 0.01

    def __init__(
        self,
        n_balls: int = 5,
        *,
        speed: float = 0.5,
        diameter: float = 0.1,
        gap: float = 0.001,
        length: float = 1.0,
        g: float = 9.81,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_balls = n_balls
        self.speed = speed
        self.diameter = diameter
        self.gap = gap
        self.length = length
        self.g = g

    def _build_system(self) -> CompositeSystem:
        """Return Newton's cradle.

        Pendulum balls in a row pass momentum along by elastic collisions.
        """
        spacing = self.diameter + self.gap
        elements = [
            HangingBallElement(
                f'ball_{index}',
                x=index * spacing,
                x_d1=self.speed if index == 0 else 0.0,
                rest=index * spacing,
                g=self.g,
                length=self.length,
            )
            for index in range(self.n_balls)
        ]
        swings: list[BaseInteraction] = [
            GradientInteraction(
                f'swing_{index}',
                target=f'ball_{index}',
                potential=lambda s: s.target.g * (s.target.x - s.target.rest) ** 2 / (2 * s.target.length),
            )
            for index in range(self.n_balls)
        ]
        collisions: list[BaseInteraction] = [
            ElasticCollisionInteraction(
                f'collision_{index}',
                left=f'ball_{index}',
                right=f'ball_{index + 1}',
                diameter=self.diameter,
            )
            for index in range(self.n_balls - 1)
        ]
        return CompositeSystem(self.label, elements=elements, interactions=[*swings, *collisions])


class CircularBilliardSystem(BaseSystem):
    """A particle moving freely inside a circle and reflecting specularly off its wall.

    The angle of incidence is the same at every bounce, so the circular billiard is integrable. The system has the
    position `particle.x`, `particle.y` and the velocity `particle.vx`, `particle.vy`.

    Args:
        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

        radius:
            The radius of the table.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CircularBilliardSystem
        >>> CircularBilliardSystem(radius=2.0).prepare().system_.elements[0].radius
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'events', 'billiard', 'integrable')
    references = ('Tabachnikov 2005', 'Datseris and Parlitz 2022')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.3,
        y: float = 0.0,
        vx: float = 0.4,
        vy: float = 1.0,
        *,
        radius: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def _build_system(self) -> CompositeSystem:
        """Return a particle moving freely inside a circle.

        The particle reflects specularly off the wall of the circle.
        """
        wall = ReflectionInteraction(
            'bounce',
            target='particle',
            guard=lambda s: s.target.x**2 + s.target.y**2 - s.target.radius**2,
            positions=('x', 'y'),
            velocities=('vx', 'vy'),
            direction=1,
        )
        return CompositeSystem(
            self.label,
            elements=[DiscBilliardElement('particle', x=self.x, y=self.y, vx=self.vx, vy=self.vy, radius=self.radius)],
            interactions=[wall],
        )
