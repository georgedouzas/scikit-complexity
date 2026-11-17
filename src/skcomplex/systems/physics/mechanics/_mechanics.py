"""Mechanics systems.

The module covers pendulums, oscillators, projectiles, and collisions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, cos, oo
from sage.sets.real_set import RealSet

from ....core import systems
from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import EventLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role, VisualSpec
from ..common.mechanics import LagrangianInteraction
from ..common.particles import LinearDragInteraction, RestoringForceInteraction
from ._springs import PeriodicForceInteraction

WALL_POINTS = 200
GRAVITY = 9.81


class BouncingBallElement(Element):
    """Ball at the height `h` above a floor.

    The height `h` and its derivative `h'`, the vertical velocity, change under the laws of gravity and of the
    impacts.

    Args:
        label:
            The label of the ball.

        h:
            The initial height.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BouncingBallElement
        >>> BouncingBallElement('ball', h=2.0).h
        2.0
    """

    def __init__(self, label: str, *, h: float = 1.0, values: Mapping[str, object] | None = None) -> None:
        super().__init__(label, values=values)
        self.h = h


def _get_disc_wall(radius: float, **_: float) -> list[np.ndarray]:
    """Return the wall of a disc table as one closed polyline."""
    angles = np.linspace(0.0, 2 * np.pi, WALL_POINTS)
    return [radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class DiscBilliardElement(Element):
    """Particle moving freely in the plane inside a disc.

    The particle has the position `(x, y)` and the velocity `(vx, vy)`, and the disc the radius `radius`.

    Args:
        label:
            The label of the particle.

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

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _get_disc_wall})

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.3,
        y: float = 0.0,
        vx: float = 0.4,
        vy: float = 1.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius


class HangingBallElement(Element):
    """Ball hanging on a string of length `length`.

    The ball swings with small amplitude about its rest position.

    The horizontal position is `x` and its derivative `x'` is the velocity. For small amplitudes the ball moves in the
    potential `g (x - rest)² / (2 length)`.

    Args:
        label:
            The label of the ball.

        x:
            The initial horizontal position.

        rest:
            The rest position.

        g:
            The gravitational acceleration.

        length:
            The length of the string.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import HangingBallElement
        >>> HangingBallElement('ball', rest=0.5).rest
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.0,
        rest: float = 0.0,
        g: float = 9.81,
        length: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
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

    law_classes = (EventLaw,)

    def __init__(
        self,
        label: str,
        *,
        left: Role,
        right: Role,
        diameter: Annotated[float, RealSet.open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.diameter = diameter

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event of the contact, which swaps the velocities of the balls.

        Args:
            context:
                The namespace of the two balls, the time `t`, and the diameter.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, left, right = context.t, context.left.x, context.right.x
            swap = {left.diff(t): right.diff(t), right.diff(t): left.diff(t)}
            self.laws_ = [EventLaw(right - left - context.diameter, swap, direction=-1)]


class HangingSwingInteraction(Interaction):
    """Small swing of a hanging ball about its rest position.

    The position `x` of the target follows `x'' = -g (x - rest) / length`, with the gravity `g`, the rest position
    `rest`, and the length `length` of the string, all values of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ball.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import HangingSwingInteraction
        >>> HangingSwingInteraction('swing', target='ball').target
        'ball'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the ball.

        Args:
            context:
                The namespace of the ball and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            ball = context.target
            self.laws_ = [RateLaw(ball.x.diff(context.t, 2), -ball.g * (ball.x - ball.rest) / ball.length)]


class UniformGravityInteraction(Interaction):
    """Uniform gravity, the constant acceleration `-g` of the height of a target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the body.

        g:
            The gravitational acceleration.

        coordinate:
            The name of the height of the body, such as `'y'` in a plane or `'h'` above a floor.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import UniformGravityInteraction
        >>> UniformGravityInteraction('gravity', target='ball', g=1.62).g
        1.62
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, g: float = GRAVITY, coordinate: str = 'y') -> None:
        super().__init__(label)
        self.target = target
        self.g = g
        self.coordinate = coordinate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the height of the body.

        Args:
            context:
                The namespace of the body, the time `t`, and the gravity.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            height = getattr(context.target, self.coordinate)
            self.laws_ = [RateLaw(height.diff(context.t, 2), -context.g)]


class FloorImpactInteraction(Interaction):
    """Impact of a ball on the floor `h = 0`, which reverses its vertical velocity and scales it by `e`.

    When the height `h` of the target falls to zero, the velocity `h'` becomes `-e h'`, with the coefficient of
    restitution `e`. It is a bouncing ball, a dropped phone, or a hopping robot.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ball, whose height is `h`.

        e:
            The coefficient of restitution, the ratio of the speeds after and before the impact.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FloorImpactInteraction
        >>> FloorImpactInteraction('impact', target='ball', e=0.5).e
        0.5
    """

    law_classes = (EventLaw,)

    def __init__(self, label: str, *, target: Role, e: Annotated[float, RealSet.closed(0, 1)] = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.e = e

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event of the impact, which reverses the vertical velocity.

        Args:
            context:
                The namespace of the ball, the time `t`, and the restitution.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            height = context.target.h
            velocity = height.diff(context.t)
            reset = {velocity: velocity - (1 + context.e) * velocity}
            self.laws_ = [EventLaw(height, reset, direction=-1, label=self.label)]


class FreeFlightInteraction(Interaction):
    """Free flight of a particle in the plane, `x' = vx` and `y' = vy`.

    The velocity `(vx, vy)` changes only at the walls, so the particle moves along straight lines between them.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FreeFlightInteraction
        >>> FreeFlightInteraction('flight', target='particle').target
        'particle'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the position of the particle.

        Args:
            context:
                The namespace of the particle and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            p, t = context.target, context.t
            self.laws_ = [RateLaw(p.x.diff(t), p.vx), RateLaw(p.y.diff(t), p.vy)]


class DiscWallInteraction(Interaction):
    """Specular reflection of a particle off the wall of a disc of radius `radius`.

    When the particle reaches the wall `x² + y² = radius²` from inside, its velocity `v = (vx, vy)` becomes
    `v - 2 (v . n) n / |n|²` with the normal `n = (2 x, 2 y)`, so the angle of incidence equals the angle of
    reflection. It is a circular billiard table, a gas in a round container, or light in a circular mirror.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle, whose values are `x`, `y`, `vx`, `vy`, and `radius`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DiscWallInteraction
        >>> DiscWallInteraction('bounce', target='particle').target
        'particle'
    """

    law_classes = (EventLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event that reflects the velocity about the normal of the wall.

        Args:
            context:
                The namespace of the particle and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            p = context.target
            guard = p.x**2 + p.y**2 - p.radius**2
            normal = [2 * p.x, 2 * p.y]
            velocities = [p.vx, p.vy]
            speed = sum((v * n for v, n in zip(velocities, normal, strict=True)), SR(0))
            scale = 2 * speed / sum((n**2 for n in normal), SR(0))
            reset = {v: v - scale * n for v, n in zip(velocities, normal, strict=True)}
            self.laws_ = [EventLaw(guard, reset, direction=1, label=self.label)]


class PendulumSwingInteraction(LagrangianInteraction):
    """Motion of a pendulum, the Euler-Lagrange equation of `L = m l² θ'² / 2 + m g l cos θ`.

    The equation is Newton's law `m l² θ'' = -m g l sin θ` for the angle `θ` of the target, whose mass `m`, length
    `l`, and gravity `g` are values of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import PendulumSwingInteraction
        >>> PendulumSwingInteraction('swing', target='bob').target
        'bob'
    """

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the angle of the pendulum."""
        return [context.target.theta]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian `m l² θ'² / 2 + m g l cos θ` of the pendulum.

        Args:
            context:
                The namespace of the pendulum and the time `t`.

            positions:
                The symbol of the angle.

            velocities:
                The symbol of the angular velocity.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        bob = context.target
        return bob.m * bob.l**2 * velocities[0] ** 2 / 2 + bob.m * bob.g * bob.l * cos(positions[0])


class ProjectileSystem(systems.BaseSystem):
    """A particle launched under uniform gravity, optionally slowed by linear drag.

    The system has the particle `ball` at the position `(ball.x, ball.y)`, with the height `ball.y`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ProjectileSystem
        >>> trajectory = ProjectileSystem(speed=5.0, angle=90.0).prepare().simulate(until=0.1)
        >>> float(trajectory.get_values("ball.x'")[0]), float(trajectory.get_values("ball.y'")[0])
        (0.0, 5.0)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.speed = speed
        self.angle = angle
        self.mass = mass
        self.drag = drag

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle launched under uniform gravity, optionally slowed by linear drag.

        The drag acts even when its coefficient is zero, so that the horizontal position moves as a state.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        radians = np.radians(self.angle)
        speeds = [round(float(self.speed * trig(radians)), 12) for trig in (np.cos, np.sin)]
        interactions = [
            UniformGravityInteraction('gravity', target='ball'),
            LinearDragInteraction('drag', target='ball', gamma=self.drag, coordinates=('x', 'y')),
        ]
        self.system_ = systems.System(
            self.label,
            elements=[Element('ball', values={'x': 0.0, 'y': 0.0, "x'": speeds[0], "y'": speeds[1], 'm': self.mass})],
            interactions=interactions,
        )


class HarmonicOscillatorSystem(systems.BaseSystem):
    """A particle on a spring, optionally damped.

    The system has the particle `mass` in one dimension, with the displacement `mass.x` and the velocity `mass.x'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.k = k
        self.mass = mass
        self.damping = damping

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle on a spring, optionally damped.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        interactions: list[BaseInteraction] = [RestoringForceInteraction('spring', target='mass', k=self.k)]
        if self.damping:
            interactions.append(LinearDragInteraction('damping', target='mass', gamma=self.damping))
        mass = Element('mass', values={'x': self.x, "x'": self.v, 'm': self.mass})
        self.system_ = systems.System(self.label, elements=[mass], interactions=interactions)


class PendulumSystem(systems.BaseSystem):
    """A pendulum, whose angle follows Newton's second law `m l θ'' = -m g sin θ`.

    The system has the pendulum `bob`, with the angle `bob.theta` and the angular velocity `bob.theta'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import PendulumSystem
        >>> trajectory = PendulumSystem(theta=0.1).prepare().simulate(until=1.0, dt=1.0)
        >>> round(float(trajectory.get_values('bob.theta')[0]), 1)
        0.1
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.theta = theta
        self.omega = omega
        self.m = m
        self.g = g
        self.l = l

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a pendulum, whose angle follows Newton's second law `m l θ'' = -m g sin θ`.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        bob = Element('bob', values={'theta': self.theta, "theta'": self.omega, 'm': self.m, 'g': self.g, 'l': self.l})
        self.system_ = systems.System(
            self.label,
            elements=[bob],
            interactions=[PendulumSwingInteraction('swing', target='bob')],
        )


class DrivenDampedOscillatorSystem(systems.BaseSystem):
    """A mass on a spring with damping and a periodic force.

    The mass follows `m x'' = -k x - c x' + F cos(ω t)`.

    The system has the mass `mass` in one dimension, with the displacement `mass.x` and the velocity `mass.x'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.F = F
        self.omega = omega

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a mass on a spring with damping and a periodic force.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = systems.System(
            self.label,
            elements=[Element('mass', values={'x': self.x, "x'": self.v, 'm': self.m})],
            interactions=[
                RestoringForceInteraction('spring', target='mass', k=self.k),
                LinearDragInteraction('damping', target='mass', gamma=self.c),
                PeriodicForceInteraction('drive', target='mass', F=self.F, omega=self.omega),
            ],
        )


class BouncingBallSystem(systems.BaseSystem):
    """A ball that falls under gravity and bounces off the floor.

    The ball loses speed by a coefficient of restitution.

    Every impact is an event that reverses the velocity and scales it by `e`. The impacts accumulate at the Zeno time
    `sqrt(2 h / g) (1 + e) / (1 - e)` for a ball dropped from rest, so runs should end before it. The system has the
    height `ball.h` and its velocity `ball.h'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BouncingBallSystem
        >>> trajectory = BouncingBallSystem().prepare().simulate(until=1.0, dt=0.01)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.h = h
        self.v = v
        self.g = g
        self.e = e

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ball that falls under gravity and bounces off the floor.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = systems.System(
            self.label,
            elements=[BouncingBallElement('ball', h=self.h, values={"h'": self.v})],
            interactions=[
                UniformGravityInteraction('gravity', target='ball', g=self.g, coordinate='h'),
                FloorImpactInteraction('impact', target='ball', e=self.e),
            ],
        )


class NewtonsCradleSystem(systems.BaseSystem):
    """Newton's cradle.

    Pendulum balls in a row pass momentum along by elastic collisions.

    The balls swing with small amplitude about rest positions that leave a small gap between neighbours, so collisions
    are separate events. The first ball starts with a speed towards the others, and equal masses exchange velocities at
    every collision. The system has the horizontal positions `ball_<i>.x` and velocities `ball_<i>.x'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_balls = n_balls
        self.speed = speed
        self.diameter = diameter
        self.gap = gap
        self.length = length
        self.g = g

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of pendulum balls in a row that pass momentum along by elastic collisions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.diameter + self.gap
        elements = [
            HangingBallElement(
                f'ball_{index}',
                x=index * spacing,
                rest=index * spacing,
                g=self.g,
                length=self.length,
                values={"x'": self.speed if index == 0 else 0.0},
            )
            for index in range(self.n_balls)
        ]
        swings: list[BaseInteraction] = [
            HangingSwingInteraction(f'swing_{index}', target=f'ball_{index}') for index in range(self.n_balls)
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
        self.system_ = systems.System(self.label, elements=elements, interactions=[*swings, *collisions])


class CircularBilliardSystem(systems.BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle moving freely inside a circle and reflecting off its wall.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        particle = DiscBilliardElement('particle', x=self.x, y=self.y, vx=self.vx, vy=self.vy, radius=self.radius)
        self.system_ = systems.System(
            self.label,
            elements=[particle],
            interactions=[
                FreeFlightInteraction('flight', target='particle'),
                DiscWallInteraction('bounce', target='particle'),
            ],
        )
