"""Mechanics models: pendulums, oscillators, projectiles, and collisions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element, ParticleElement
from ....core.interactions import BaseInteraction, Interaction
from ....core.laws import BaseLaw, CrossingLaw, EquationLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.particles import ConstantForceInteraction, HarmonicPotentialInteraction, LinearDragInteraction

GRAVITY = 9.81
WALL_POINTS = 200


def _pendulum_bob(theta: float, l: float, **_: float) -> list[list[float]]:
    """Return the position of the bob of a pendulum hanging from the origin."""
    return [[l * np.sin(theta), -l * np.cos(theta)]]


def _disc_wall(radius: float, **_: float) -> list[np.ndarray]:
    """Return the wall of a disc table as one closed polyline."""
    angles = np.linspace(0.0, 2 * np.pi, WALL_POINTS)
    return [radius * np.column_stack([np.cos(angles), np.sin(angles)])]


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode'),
    references=('Newton 1687',),
    defaults={'until': 2.0, 'dt': 0.01},
)
def projectile(speed: float = 10.0, angle: float = 45.0, *, mass: float = 1.0, drag: float = 0.0) -> System:
    """Return a particle launched under uniform gravity, optionally slowed by linear drag.

    Args:
        speed:
            The launch speed.

        angle:
            The launch angle above the horizon, in degrees.

        mass:
            The mass.

        drag:
            The linear drag coefficient.

    Returns:
        system:
            The model, with the particle `ball` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.mechanics import projectile
        >>> projectile(speed=5.0, angle=90.0).elements[0].v[1]
        5.0
    """
    radians = np.radians(angle)
    velocity = [round(float(speed * np.cos(radians)), 12), round(float(speed * np.sin(radians)), 12)]
    interactions: list[BaseInteraction] = [ConstantForceInteraction('gravity', target='ball', F=[0.0, -GRAVITY * mass])]
    if drag:
        interactions.append(LinearDragInteraction('drag', target='ball', gamma=drag))
    return System(
        elements=[ParticleElement('ball', x=[0.0, 0.0], v=velocity, m=mass)],
        interactions=interactions,
        space=ContinuousSpace(n_dim=2),
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'oscillation'),
    references=('Hooke 1678',),
    defaults={'until': 20.0, 'dt': 0.05},
)
def harmonic_oscillator(
    x: float = 1.0,
    v: float = 0.0,
    *,
    k: float = 1.0,
    mass: float = 1.0,
    damping: float = 0.0,
) -> System:
    """Return a particle on a spring, optionally damped.

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

    Returns:
        system:
            The model, with the particle `mass` in one dimension.

    Examples:
        >>> from skcomplex.models.physics.mechanics import harmonic_oscillator
        >>> harmonic_oscillator(k=4.0).interactions[0].k
        4.0
    """
    interactions: list[BaseInteraction] = [HarmonicPotentialInteraction('spring', target='mass', k=k)]
    if damping:
        interactions.append(LinearDragInteraction('damping', target='mass', gamma=damping))
    return System(
        elements=[ParticleElement('mass', x=x, v=v, m=mass)],
        interactions=interactions,
        space=ContinuousSpace(n_dim=1),
    )


class PendulumElement(Element):
    """Pendulum element."""

    visual = VisualSpec('bodies', {'positions': _pendulum_bob, 'pivot': (0.0, 0.0), 'rods': True})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the pendulum element."""
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]


class OscillatorElement(Element):
    """Oscillator element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        F: Annotated[object, ParameterSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.m = m
        self.k = k
        self.c = c
        self.F = F
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the oscillator element."""
        return [RateLaw(s.x, (-s.k * s.x - s.c * derivative(s.x, 1) + s.F * cos(s.omega * s.t)) / s.m)]


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'oscillation'),
    references=('Newton 1687',),
    defaults={'until': 10.0, 'dt': 0.01},
)
def pendulum(
    theta: float = 0.5,
    omega: float = 0.0,
    *,
    m: float = 1.0,
    g: float = 9.81,
    l: float = 1.0,
) -> System:
    """Return a pendulum written as Newton's second law, `m l θ'' = -m g sin θ`.

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

    Returns:
        system:
            The model, with the pendulum `bob`, whose angle has order 2 and angular velocity `bob.theta_d1`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import pendulum
        >>> pendulum(theta=0.1).elements[0].theta_d1
        0.0
    """
    return System(elements=[PendulumElement('bob', theta=theta, theta_d1=omega, m=m, g=g, l=l)])


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'oscillation', 'forcing'),
    references=('Hooke 1678',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def driven_damped_oscillator(
    x: float = 1.0,
    v: float = 0.0,
    *,
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.1,
    F: float = 0.0,
    omega: float = 1.0,
) -> System:
    """Return a mass on a spring with damping and a periodic force, `m x'' = -k x - c x' + F cos(ω t)`.

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

    Returns:
        system:
            The model, with the oscillator `mass`, whose displacement has order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import driven_damped_oscillator
        >>> driven_damped_oscillator(F=0.5).elements[0].F
        0.5
    """
    return System(elements=[OscillatorElement('mass', x=x, x_d1=v, m=m, k=k, c=c, F=F, omega=omega)])


class BouncingBallElement(Element):
    """Bouncing ball element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[object, StateSpec(order=2)] = 1.0,
        h_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        e: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.h_d1 = h_d1
        self.g = g
        self.e = e

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the bouncing ball element."""
        return [
            RateLaw(s.h, -s.g),
            CrossingLaw(s.h, {derivative(s.h, 1): -s.e * derivative(s.h, 1)}, direction=-1, label='impact'),
        ]


class HangingBallElement(Element):
    """Hanging ball element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        rest: Annotated[object, ParameterSpec()] = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        length: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.rest = rest
        self.g = g
        self.length = length

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the hanging ball element."""
        return [RateLaw(s.x, -s.g / s.length * (s.x - s.rest))]


class ElasticCollisionInteraction(Interaction):
    """Elastic collision interaction."""

    def __init__(
        self,
        label: str,
        *,
        left: Annotated[object, RoleSpec()],
        right: Annotated[object, RoleSpec()],
        diameter: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.diameter = diameter

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the elastic collision interaction."""
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


class DiscBilliardElement(Element):
    """Disc billiard element."""

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _disc_wall})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.3,
        y: Annotated[object, StateSpec()] = 0.0,
        vx: Annotated[object, StateSpec()] = 0.4,
        vy: Annotated[object, StateSpec()] = 1.0,
        radius: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the disc billiard element."""
        return [
            RateLaw(s.x, s.vx),
            RateLaw(s.y, s.vy),
            CrossingLaw(
                s.x**2 + s.y**2 - s.radius**2,
                {
                    s.vx: s.vx - 2 * (s.vx * s.x + s.vy * s.y) * s.x / (s.x**2 + s.y**2),
                    s.vy: s.vy - 2 * (s.vx * s.x + s.vy * s.y) * s.y / (s.x**2 + s.y**2),
                },
                direction=1,
                label='bounce',
            ),
        ]


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'events', 'impacts'),
    references=('Strogatz 2015', 'Hairer, Nørsett, and Wanner 1993'),
    defaults={'until': 3.0, 'dt': 0.01},
)
def bouncing_ball(h: float = 1.0, v: float = 0.0, *, g: float = 9.81, e: float = 0.8) -> System:
    """Return a ball that falls under gravity and bounces off the floor, losing speed by a coefficient of restitution.

    Every impact is an event that reverses the velocity and scales it by `e`. The impacts accumulate at the Zeno time
    `sqrt(2 h / g) (1 + e) / (1 - e)` for a ball dropped from rest, so runs should end before it.

    Args:
        h:
            The initial height.

        v:
            The initial vertical velocity.

        g:
            The gravitational acceleration.

        e:
            The coefficient of restitution.

    Returns:
        system:
            The model, with the height `ball.h` of order 2 and its velocity `ball.h_d1`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import bouncing_ball
        >>> from skcomplex.engine.runtime import Simulation
        >>> trajectory = Simulation(bouncing_ball(), dt=0.01).check().prepare().run(until=1.0).trajectory_
        >>> round(trajectory.events[0].time, 6)
        0.451524
    """
    return System(elements=[BouncingBallElement('ball', h=h, h_d1=v, g=g, e=e)])


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'events', 'impacts', 'momentum'),
    references=('Hinch and Saint-Jean 1999',),
    defaults={'until': 5.0, 'dt': 0.01},
)
def newtons_cradle(
    n_balls: int = 5,
    *,
    speed: float = 0.5,
    diameter: float = 0.1,
    gap: float = 0.001,
    length: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return Newton's cradle: pendulum balls in a row that pass momentum along by elastic collisions.

    The balls swing with small amplitude about rest positions that leave a small gap between neighbours, so collisions
    are separate events. The first ball starts with a speed towards the others, and equal masses exchange velocities
    at every collision.

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

    Returns:
        system:
            The model, with the horizontal positions `ball_<i>.x` and velocities `ball_<i>.x_d1`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import newtons_cradle
        >>> len(newtons_cradle(n_balls=3).interactions)
        2
    """
    spacing = diameter + gap
    elements = [
        HangingBallElement(
            f'ball_{index}',
            x=index * spacing,
            x_d1=speed if index == 0 else 0.0,
            rest=index * spacing,
            g=g,
            length=length,
        )
        for index in range(n_balls)
    ]
    interactions = [
        ElasticCollisionInteraction(
            f'collision_{index}',
            left=f'ball_{index}',
            right=f'ball_{index + 1}',
            diameter=diameter,
        )
        for index in range(n_balls - 1)
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'events', 'billiard', 'integrable'),
    references=('Tabachnikov 2005', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def circular_billiard(
    x: float = 0.3,
    y: float = 0.0,
    vx: float = 0.4,
    vy: float = 1.0,
    *,
    radius: float = 1.0,
) -> System:
    """Return a particle moving freely inside a circle and reflecting specularly off its wall.

    The angle of incidence is the same at every bounce, so the circular billiard is integrable.

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

    Returns:
        system:
            The model, with the position `particle.x`, `particle.y` and the velocity `particle.vx`, `particle.vy`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import circular_billiard
        >>> circular_billiard(radius=2.0).elements[0].radius
        2.0
    """
    return System(elements=[DiscBilliardElement('particle', x=x, y=y, vx=vx, vy=vy, radius=radius)])
