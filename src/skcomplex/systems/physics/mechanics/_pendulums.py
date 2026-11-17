"""Pendulum systems.

The module covers double, coupled, parametric, magnetic, and ballistic pendulums, carts, and clocks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sign, sin
from sage.sets.real_set import RealSet

from ....core import systems
from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import EventLaw, RateLaw
from ....core.components.laws.utils import derivative
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, Role, RoleSpec, StateSpec, VisualSpec
from ...general.mechanics import EulerLagrangeInteraction
from ..common.mechanics import LagrangianInteraction, PendulumElement, PendulumInteraction
from ..common.particles import LinearDragInteraction, RestoringForceInteraction

DIRECTIONS = ('vertical', 'horizontal')


class BallisticPendulumElement(Element):
    """Bullet of mass `m` at the position `x` flying towards the block of a pendulum.

    The block has the mass `M`, and the pendulum the angle `theta`.

    The bullet flies freely, the pendulum swings, and the impact transfers the momentum of the bullet to the pendulum.
    The velocities are `x'` and `theta'`.

    Args:
        label:
            The label of the bullet and pendulum.

        x:
            The initial position of the bullet, negative before the block.

        theta:
            The initial angle of the pendulum.

        m:
            The mass of the bullet.

        M:
            The mass of the block.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BallisticPendulumElement
        >>> BallisticPendulumElement('pendulum', M=2.0).M
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = -1.0,
        theta: float = 0.0,
        m: Annotated[float, RealSet.open(0, oo)] = 0.01,
        M: Annotated[float, RealSet.open(0, oo)] = 1.0,
        l: Annotated[float, RealSet.open(0, oo)] = 1.0,
        g: Annotated[float, RealSet.open(0, oo)] = 9.81,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
        self.theta = theta
        self.m = m
        self.M = M
        self.l = l
        self.g = g


def _get_cart_and_bob(y: float, theta: float, l: float, **_: float) -> np.ndarray:
    """Return the positions of a cart on a horizontal track and of its hanging bob."""
    return np.array([[y, 0.0], [y + l * np.sin(theta), -l * np.cos(theta)]])


class CartPendulumElement(Element):
    """Cart on a horizontal track at the position `y`.

    A pendulum of angle `theta` hangs from the cart.

    The position `y` and the angle `theta` are the coordinates of `CartPendulumInteraction`, and their velocities are
    `y'` and `theta'`.

    Args:
        label:
            The label of the cart and pendulum.

        y:
            The initial position of the cart.

        theta:
            The initial angle of the pendulum.

        m1:
            The mass of the cart.

        m2:
            The mass of the bob.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CartPendulumElement
        >>> CartPendulumElement('cart', m1=5.0).m1
        5.0
    """

    visual = VisualSpec('bodies', {'positions': _get_cart_and_bob, 'rods': True, 'traced': [1]})

    def __init__(
        self,
        label: str,
        *,
        y: float = 0.0,
        theta: float = 0.5,
        m1: Annotated[float, RealSet.open(0, oo)] = 1.0,
        m2: Annotated[float, RealSet.open(0, oo)] = 1.0,
        l: Annotated[float, RealSet.open(0, oo)] = 1.0,
        g: Annotated[float, RealSet.open(0, oo)] = 9.81,
    ) -> None:
        super().__init__(label)
        self.y = y
        self.theta = theta
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g


class ClockPendulumElement(Element):
    """Pendulum clock, a pendulum of angle `theta` hanging from a support.

    The support is at the horizontal position `x`.

    The angle and the position have order 2. The motion comes from `ClockPendulumInteraction`.

    Args:
        label:
            The label of the clock.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

        x:
            The initial position of the support.

        x_d1:
            The initial velocity of the support.

        m:
            The mass of the bob.

        l:
            The length of the pendulum.

        M:
            The mass of the support.

        g:
            The gravitational acceleration.

        rho:
            The damping of the support.

        lam:
            The strength of the escapement.

        theta_c:
            The angle below which the escapement pumps energy in.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ClockPendulumElement
        >>> ClockPendulumElement('clock', lam=0.2).lam
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float, StateSpec(order=2)] = 0.2,
        theta_d1: float = 0.0,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.8,
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        theta_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.x = x
        self.x_d1 = x_d1
        self.m = m
        self.l = l
        self.M = M
        self.g = g
        self.rho = rho
        self.lam = lam
        self.theta_c = theta_c


def _get_double_bobs(theta1: float, theta2: float, l1: float, l2: float, **_: float) -> np.ndarray:
    """Return the positions of the two bobs of a double pendulum.

    The pendulum hangs from the origin.
    """
    first = np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)])
    return np.stack([first, first + np.array([l2 * np.sin(theta2), -l2 * np.cos(theta2)])])


class DoublePendulumElement(Element):
    """Double pendulum.

    A bob of mass `m2` on a rod of length `l2` hangs from a bob of mass `m1` on a rod `l1`.

    The angles `theta1` and `theta2` of the rods from the vertical have order 2. The motion comes from
    `DoublePendulumInteraction`.

    Args:
        label:
            The label of the double pendulum.

        theta1:
            The initial angle of the upper rod.

        theta1_d1:
            The initial angular velocity of the upper rod.

        theta2:
            The initial angle of the lower rod.

        theta2_d1:
            The initial angular velocity of the lower rod.

        m1:
            The mass of the upper bob.

        m2:
            The mass of the lower bob.

        l1:
            The length of the upper rod.

        l2:
            The length of the lower rod.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DoublePendulumElement
        >>> DoublePendulumElement('pendulum', m2=2.0).m2
        2.0
    """

    visual = VisualSpec(
        'bodies',
        {'positions': _get_double_bobs, 'pivot': (0.0, 0.0), 'rods': True, 'traced': [1]},
    )

    def __init__(
        self,
        label: str,
        *,
        theta1: Annotated[float, StateSpec(period=2 * pi, order=2)] = math.pi / 2,
        theta1_d1: float = 0.0,
        theta2: Annotated[float, StateSpec(period=2 * pi, order=2)] = math.pi / 2,
        theta2_d1: float = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
    ) -> None:
        super().__init__(label)
        self.theta1 = theta1
        self.theta1_d1 = theta1_d1
        self.theta2 = theta2
        self.theta2_d1 = theta2_d1
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g


class ParametricPendulumElement(Element):
    """Pendulum whose suspension point oscillates.

    The angle `theta` is the coordinate of `ParametricPendulumInteraction`, and `theta'` is its angular velocity.

    Args:
        label:
            The label of the pendulum.

        theta:
            The initial angle.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumElement
        >>> ParametricPendulumElement('swing', l=0.5).l
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        theta: float = 0.1,
        g: Annotated[float, RealSet.open(0, oo)] = 9.81,
        l: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.theta = theta
        self.g = g
        self.l = l


class CartPendulumInteraction(LagrangianInteraction):
    """Motion of a cart and the pendulum that hangs from it.

    The motion is the Euler-Lagrange equations with a force on the cart.

    The Lagrangian is `(m1 + m2) y'² / 2 + m2 l y' θ' cos θ + m2 l² θ'² / 2 + m2 g l cos θ`, and the force `F` drives
    the position `y` of the cart.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the cart and pendulum.

        F:
            A constant horizontal force on the cart.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CartPendulumInteraction
        >>> CartPendulumInteraction('motion', target='cart', F=1.0).F
        1.0
    """

    def __init__(self, label: str, *, target: Role, F: float = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.F = F

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the position of the cart and the angle of the pendulum."""
        return [context.target.y, context.target.theta]

    def _get_forces(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> list[Expression]:
        """Return the generalized forces, the force `F` on the cart and none on the angle."""
        return [context.F, 0]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian of the cart and pendulum.

        Args:
            context:
                The namespace of the cart, the time `t`, and the force.

            positions:
                The symbols of the position and the angle.

            velocities:
                The symbols of the velocity and the angular velocity.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = context.target
        theta = positions[1]
        v, w = velocities
        kinetic = (p.m1 + p.m2) * v**2 / 2 + p.m2 * p.l * v * w * cos(theta) + p.m2 * p.l**2 * w**2 / 2
        return kinetic + p.m2 * p.g * p.l * cos(theta)


class ClockPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a pendulum clock on a moving support.

    The motion is the Euler-Lagrange equations with the escapement.

    The Lagrangian is `(M + m) x'² / 2 + m l x' θ' cos θ + m l² θ'² / 2 + m g l cos θ`. The Rayleigh dissipation
    `λ l sgn(|θ| - θ_c) θ'² + rho x'²` damps the support and gives the escapement `2 λ l sgn(|θ| - θ_c) θ'`, which pumps
    energy into the pendulum below the angle `θ_c` and takes it out beyond.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the clock.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ClockPendulumInteraction
        >>> ClockPendulumInteraction('motion', target='clock').states
        ('theta', 'x')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('theta', 'x'), dissipation=self._get_escapement)

    @staticmethod
    def _get_escapement(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation of the escapement and of the support damping."""
        p = namespace.target
        escapement = p.lam * p.l * sign(abs(p.theta) - p.theta_c) * derivative(p.theta, 1) ** 2
        return escapement + p.rho * derivative(p.x, 1) ** 2

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the pendulum and its support.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        v, w = derivative(p.x, 1), derivative(p.theta, 1)
        kinetic = (p.M + p.m) * v**2 / 2 + p.m * p.l * v * w * cos(p.theta) + p.m * p.l**2 * w**2 / 2
        return kinetic + p.m * p.g * p.l * cos(p.theta)


class DoublePendulumInteraction(EulerLagrangeInteraction):
    """Motion of a double pendulum.

    The motion is the Euler-Lagrange equations of its kinetic energy and gravity.

    The Lagrangian is `(m1 + m2) l1² θ1'² / 2 + m2 l2² θ2'² / 2 + m2 l1 l2 θ1' θ2' cos(θ1 - θ2) + (m1 + m2) g l1 cos θ1
    + m2 g l2 cos θ2`, with the masses, lengths, and gravity of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the double pendulum.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DoublePendulumInteraction
        >>> DoublePendulumInteraction('motion', target='pendulum').states
        ('theta1', 'theta2')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('theta1', 'theta2'))

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the double pendulum.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        w1, w2 = derivative(p.theta1, 1), derivative(p.theta2, 1)
        kinetic = (p.m1 + p.m2) * p.l1**2 * w1**2 / 2 + p.m2 * p.l2**2 * w2**2 / 2
        kinetic += p.m2 * p.l1 * p.l2 * w1 * w2 * cos(p.theta1 - p.theta2)
        return kinetic + (p.m1 + p.m2) * p.g * p.l1 * cos(p.theta1) + p.m2 * p.g * p.l2 * cos(p.theta2)


class MagnetInteraction(Interaction):
    """Magnet below the plane of a pendulum bob.

    The magnet at `(x0, y0)` attracts the bob of mass `m` with the energy `-gamma / sqrt(r² + d²)`, where `r` is the
    distance of the bob from the magnet in the plane, so the bob follows `m x'' = -gamma (x - x0) / (r² + d²)^(3/2)` and
    `m y'' = -gamma (y - y0) / (r² + d²)^(3/2)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bob.

        x0:
            The horizontal position of the magnet in the plane.

        y0:
            The vertical position of the magnet in the plane.

        gamma:
            The strength of the magnet.

        d:
            The height of the bob above the plane of the magnets.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import MagnetInteraction
        >>> MagnetInteraction('magnet', target='bob', x0=1.0).gamma
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        x0: float = 0.0,
        y0: float = 0.0,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        d: Annotated[float, RealSet.open(0, oo)] = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.x0 = x0
        self.y0 = y0
        self.gamma = gamma
        self.d = d

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the bob towards the magnet.

        Args:
            context:
                The namespace of the bob, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, bob = context.t, context.target
            dx, dy = bob.x - context.x0, bob.y - context.y0
            pull = -context.gamma / (dx**2 + dy**2 + context.d**2) ** (3 / 2) / bob.m
            self.laws_ = [RateLaw(bob.x.diff(t, 2), pull * dx), RateLaw(bob.y.diff(t, 2), pull * dy)]


class ParametricPendulumInteraction(LagrangianInteraction):
    """Motion of a pendulum whose suspension point oscillates.

    The point oscillates vertically or horizontally, and the motion is an Euler-Lagrange equation.

    The suspension point accelerates by `ε g cos Ω t`. Vertically, the Lagrangian per moment of inertia is
    `θ'² / 2 + (g / l)(1 + ε cos Ω t) cos θ`, and the acceleration modulates gravity, a parametric forcing.
    Horizontally, it is `θ'² / 2 + (g / l)(cos θ - ε cos Ω t sin θ)`, and the acceleration pushes the bob sideways, an
    external forcing. The Rayleigh dissipation `c θ'² / 2` adds the friction, so the angle follows
    `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0` or `θ'' + c θ' + (g / l)(sin θ + ε cos Ω t cos θ) = 0`, with the
    gravity `g` and the length `l` of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

        epsilon:
            The amplitude of the acceleration of the suspension point, relative to `g`.

        Omega:
            The angular frequency of the suspension point.

        c:
            The friction coefficient.

        direction:
            The direction in which the suspension point oscillates, `'vertical'` or `'horizontal'`.

    Raises:
        ValidationError:
            The direction is unknown.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumInteraction
        >>> ParametricPendulumInteraction('swing', target='pendulum', epsilon=0.5).epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        epsilon: float = 0.2,
        Omega: Annotated[float, RealSet.closed_open(0, oo)] = 2 * math.sqrt(9.81),
        c: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        direction: str = 'vertical',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.epsilon = epsilon
        self.Omega = Omega
        self.c = c
        self.direction = direction

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Raise an error for an unknown direction of the suspension point, then set the law of the angle.

        Args:
            context:
                The namespace of the pendulum, the time `t`, and the parameters.

        Raises:
            ValidationError:
                The direction is unknown.
        """
        if self.direction not in DIRECTIONS:
            error_msg = f'Unknown direction {self.direction!r}. Use one of {", ".join(DIRECTIONS)}.'
            raise ValidationError(error_msg)
        super()._prepare(context)

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the angle of the pendulum."""
        return [context.target.theta]

    def _get_dissipation(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Rayleigh dissipation `c θ'² / 2` of the friction."""
        return context.c * velocities[0] ** 2 / 2

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian of the parametric pendulum per moment of inertia.

        Args:
            context:
                The namespace of the pendulum, the time `t`, and the parameters.

            positions:
                The symbol of the angle.

            velocities:
                The symbol of the angular velocity.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = context.target
        theta = positions[0]
        drive = context.epsilon * cos(context.Omega * context.t)
        if self.direction == 'horizontal':
            return velocities[0] ** 2 / 2 + p.g / p.l * (cos(theta) - drive * sin(theta))
        return velocities[0] ** 2 / 2 + p.g / p.l * (1 + drive) * cos(theta)


class PendulumSpringInteraction(EulerLagrangeInteraction):
    """Spring between the bobs of two pendulums.

    The spring stretches with the horizontal distance of the bobs.

    The spring of stiffness `k` adds the Lagrangian `-k (l2 sin θ2 - l1 sin θ1)² / 2` to the Euler-Lagrange equations
    of the two angles, so its torque on every pendulum divides by the moment of inertia `m l²` of the pendulum's own
    Lagrangian.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first pendulum, with the angle `theta`, the mass `m`, and the length `l`.

        second:
            The path of the second pendulum.

        k:
            The stiffness of the spring.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import PendulumSpringInteraction
        >>> PendulumSpringInteraction('spring', first='left', second='right', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=first, states='theta')
        self.first = first
        self.second = second
        self.k = k

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return minus the energy of the spring.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use, with `first` and `second`.

        Returns:
            lagrangian:
                The Lagrangian of the spring.
        """
        first, second = namespace.first, namespace.second
        stretch = second.l * sin(second.theta) - first.l * sin(first.theta)
        return -namespace.k * stretch**2 / 2


class SupportSpringInteraction(EulerLagrangeInteraction):
    """Spring of stiffness `k` between the supports of two pendulum clocks.

    The spring adds the Lagrangian `-k (x2 - x1)² / 2` to the Euler-Lagrange equations of the supports, so every clock
    answers with the accelerations of its support and its pendulum through the mass matrix of its own Lagrangian.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first clock.

        second:
            The path of the second clock.

        k:
            The stiffness of the spring.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import SupportSpringInteraction
        >>> SupportSpringInteraction('spring', first='clock_0', second='clock_1', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=first, states='x')
        self.first = first
        self.second = second
        self.k = k

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return minus the energy of the spring.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use, with `first` and `second`.

        Returns:
            lagrangian:
                The Lagrangian of the spring.
        """
        return -namespace.k * (namespace.second.x - namespace.first.x) ** 2 / 2


class DoublePendulumSystem(BaseSystem):
    """A double pendulum, a pendulum hanging from the bob of another.

    The system is written as its two Lagrange equations.

    The equations `(m1 + m2) l1 θ1'' + m2 l2 θ2'' cos(θ1 - θ2) + m2 l2 θ2'² sin(θ1 - θ2) + (m1 + m2) g sin θ1 = 0` and
    `l2 θ2'' + l1 θ1'' cos(θ1 - θ2) - l1 θ1'² sin(θ1 - θ2) + g sin θ2 = 0` are solved jointly for the angular
    accelerations. The motion is chaotic at large amplitudes. The system has the angles `pendulum.theta1` and
    `pendulum.theta2` of order 2.

    Args:
        theta1:
            The initial angle of the upper rod from the vertical, in radians.

        theta2:
            The initial angle of the lower rod from the vertical, in radians.

        omega1:
            The initial angular velocity of the upper rod.

        omega2:
            The initial angular velocity of the lower rod.

        m1:
            The mass of the upper bob.

        m2:
            The mass of the lower bob.

        l1:
            The length of the upper rod.

        l2:
            The length of the lower rod.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DoublePendulumSystem
        >>> DoublePendulumSystem(m2=2.0).prepare().system_.elements[0].m2
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'chaos', 'lagrangian')
    references = ('Ryzhov et al. 2021', 'Argun et al. 2021')
    until = 20.0
    dt = 0.02

    def __init__(
        self,
        theta1: float = math.pi / 2,
        theta2: float = math.pi / 2,
        omega1: float = 0.0,
        omega2: float = 0.0,
        *,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                DoublePendulumElement(
                    'pendulum',
                    theta1=self.theta1,
                    theta2=self.theta2,
                    theta1_d1=self.omega1,
                    theta2_d1=self.omega2,
                    m1=self.m1,
                    m2=self.m2,
                    l1=self.l1,
                    l2=self.l2,
                    g=self.g,
                ),
            ],
            interactions=[DoublePendulumInteraction('motion', target='pendulum')],
        )


class CoupledPendulumsSystem(BaseSystem):
    """Two pendulums whose bobs are joined by a spring.

    The pendulums pass their energy back and forth.

    The spring stretches with the horizontal distance `l2 sin θ2 - l1 sin θ1` of the bobs and adds the torque `k l1 (l2
    sin θ2 - l1 sin θ1) cos θ1` to the first pendulum and the opposite one to the second. Identical pendulums have the
    normal modes `sqrt(g / l)` in phase and `sqrt(g / l + 2 k / m)` in antiphase. The system has the pendulums `left`
    and `right` and their angles `left.theta` and `right.theta`.

    Args:
        theta1:
            The initial angle of the first pendulum.

        theta2:
            The initial angle of the second pendulum.

        k:
            The stiffness of the spring.

        m1:
            The mass of the first bob.

        m2:
            The mass of the second bob.

        l1:
            The length of the first pendulum.

        l2:
            The length of the second pendulum.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CoupledPendulumsSystem
        >>> CoupledPendulumsSystem(k=2.0).prepare().system_.interactions[0].k
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'oscillation', 'normal modes')
    references = ('Ryzhov et al. 2021',)
    until = 40.0
    dt = 0.05

    def __init__(
        self,
        theta1: float = 0.2,
        theta2: float = 0.0,
        *,
        k: float = 0.5,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.theta1 = theta1
        self.theta2 = theta2
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def _build_system(self) -> CompositeSystem:
        """Return two pendulums whose bobs are joined by a spring.

        The pendulums pass their energy back and forth.
        """
        return CompositeSystem(
            self.label,
            elements=[
                PendulumElement('left', theta=self.theta1, m=self.m1, l=self.l1, g=self.g),
                PendulumElement('right', theta=self.theta2, m=self.m2, l=self.l2, g=self.g),
            ],
            interactions=[
                PendulumSpringInteraction('spring', first='left', second='right', k=self.k),
                PendulumInteraction('left_swing', target='left'),
                PendulumInteraction('right_swing', target='right'),
            ],
        )


class CartPendulumSystem(systems.BaseSystem):
    """A pendulum hanging from a cart.

    The cart slides without friction along a horizontal track.

    The Lagrange equations `(m1 + m2) y'' + m2 l (θ'' cos θ - θ'² sin θ) = F` and `l θ'' + y'' cos θ + g sin θ = 0`
    couple the position `y` of the cart and the angle `θ` of the pendulum. Without a force the horizontal momentum is
    conserved, so the cart swings against the pendulum. The system has the cart position `cart.y` and the angle
    `cart.theta`, and their velocities `cart.y'` and `cart.theta'`.

    Args:
        theta:
            The initial angle of the pendulum.

        y:
            The initial position of the cart.

        m1:
            The mass of the cart.

        m2:
            The mass of the bob.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

        F:
            A constant horizontal force on the cart.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import CartPendulumSystem
        >>> CartPendulumSystem(m1=5.0).prepare().system_.elements[0].m1
        5.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'lagrangian', 'momentum')
    references = ('Ryzhov et al. 2021',)
    until = 10.0
    dt = 0.02

    def __init__(
        self,
        theta: float = 0.5,
        y: float = 0.0,
        *,
        m1: float = 1.0,
        m2: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        F: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.theta = theta
        self.y = y
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.F = F

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a pendulum hanging from a cart that slides along a horizontal track.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        cart = CartPendulumElement('cart', y=self.y, theta=self.theta, m1=self.m1, m2=self.m2, l=self.l, g=self.g)
        motion = CartPendulumInteraction('motion', target='cart', F=self.F)
        self.system_ = systems.System(self.label, elements=[cart], interactions=[motion])


class ParametricPendulumSystem(systems.BaseSystem):
    """A pendulum whose suspension point oscillates vertically or horizontally.

    The suspension point has the acceleration `ε g cos Ω t`.

    Vertically, the acceleration of the suspension point modulates gravity, `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ =
    0`. Small oscillations grow when `Ω` is close to twice the natural frequency `sqrt(g / l)`, the parametric resonance
    of a swing, while a pendulum at rest stays at rest. Horizontally, the acceleration pushes the bob sideways, `θ'' + c
    θ' + (g / l)(sin θ + ε cos Ω t cos θ) = 0`, which for small angles is a harmonic oscillator under an external force.
    Its amplitude grows linearly in time, as `ε Ω t / 2`, when `Ω` equals the natural frequency, the linear resonance,
    and the pendulum starts to swing even from rest. The system has the angle `swing.theta` and the angular velocity
    `swing.theta'`.

    Args:
        theta:
            The initial angle.

        omega:
            The initial angular velocity.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

        epsilon:
            The amplitude of the acceleration of the suspension point, relative to `g`.

        Omega:
            The angular frequency of the suspension point.

        c:
            The friction coefficient.

        direction:
            The direction in which the suspension point oscillates, `'vertical'` or `'horizontal'`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import ParametricPendulumSystem
        >>> ParametricPendulumSystem(epsilon=0.5).prepare().system_.interactions[0].epsilon
        0.5
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'forcing', 'parametric resonance')
    references = ('Dilão 2023',)
    until = 30.0
    dt = 0.02

    def __init__(
        self,
        theta: float = 0.1,
        omega: float = 0.0,
        *,
        g: float = 9.81,
        l: float = 1.0,
        epsilon: float = 0.2,
        Omega: float = 2 * math.sqrt(9.81),
        c: float = 0.0,
        direction: str = 'vertical',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.theta = theta
        self.omega = omega
        self.g = g
        self.l = l
        self.epsilon = epsilon
        self.Omega = Omega
        self.c = c
        self.direction = direction

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a pendulum whose suspension point oscillates.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        motion = ParametricPendulumInteraction(
            'motion',
            target='swing',
            epsilon=self.epsilon,
            Omega=self.Omega,
            c=self.c,
            direction=self.direction,
        )
        swing = ParametricPendulumElement('swing', theta=self.theta, g=self.g, l=self.l, values={"theta'": self.omega})
        self.system_ = systems.System(self.label, elements=[swing], interactions=[motion])


class MagneticPendulumSystem(systems.BaseSystem):
    """A magnetic bob on a string swinging above magnets on a circle.

    The bob comes to rest at one of the magnets.

    The bob follows `x'' = -ω² x - q x' - Σ gamma (x - x_i) / d_i³` with `d_i = sqrt(|x - x_i|² + d²)`. The basins of
    the magnets have fractal boundaries. The system has the bob `bob` and its position `(bob.x, bob.y)`.

    Args:
        x:
            The initial horizontal position of the bob.

        y:
            The initial vertical position of the bob.

        n_magnets:
            The number of magnets, evenly spaced on the unit circle.

        omega:
            The natural frequency of the pendulum.

        q:
            The friction coefficient.

        gamma:
            The strength of every magnet.

        d:
            The height of the bob above the plane of the magnets.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import MagneticPendulumSystem
        >>> len(MagneticPendulumSystem(n_magnets=4).prepare().system_.interactions)
        6
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'multistability', 'fractal basins')
    references = ('Datseris and Parlitz 2022',)
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        y: float = 1.5,
        *,
        n_magnets: int = 3,
        omega: float = 1.0,
        q: float = 0.2,
        gamma: float = 1.0,
        d: float = 0.3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.n_magnets = n_magnets
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.d = d

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a magnetic bob on a string swinging above magnets on a circle.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        angles = 2 * np.pi * np.arange(self.n_magnets) / self.n_magnets
        magnets = [
            MagnetInteraction(
                f'magnet_{index}',
                target='bob',
                x0=round(float(np.cos(angle)), 12),
                y0=round(float(np.sin(angle)), 12),
                gamma=self.gamma,
                d=self.d,
            )
            for index, angle in enumerate(angles)
        ]
        plane = ('x', 'y')
        self.system_ = systems.System(
            self.label,
            elements=[Element('bob', values={'x': self.x, 'y': self.y, 'm': 1.0})],
            interactions=[
                RestoringForceInteraction('string', target='bob', k=self.omega**2, coordinates=plane),
                LinearDragInteraction('friction', target='bob', gamma=self.q, coordinates=plane),
                *magnets,
            ],
        )


class BlockSwingInteraction(LagrangianInteraction):
    """Swing of the block of a ballistic pendulum, the Euler-Lagrange equation of `L = l θ'² / 2 + g cos θ`.

    The angle `theta` of the target follows `l θ'' = -g sin θ`, with the length `l` and the gravity `g` of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bullet and pendulum.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BlockSwingInteraction
        >>> BlockSwingInteraction('swing', target='pendulum').target
        'pendulum'
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
        """Return the Lagrangian `l θ'² / 2 + g cos θ` of the block.

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
        return context.target.l * velocities[0] ** 2 / 2 + context.target.g * cos(positions[0])


class BulletImpactInteraction(Interaction):
    """Free flight of a bullet that sticks in the block of a pendulum.

    The position `x` of the bullet follows `x'' = 0` until it reaches the block at `x = 0`. The impact stops the bullet
    and gives the pendulum the angular velocity `m x' / ((m + M) l)`, which conserves momentum, with the masses `m` of
    the bullet and `M` of the block and the length `l` of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bullet and pendulum.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BulletImpactInteraction
        >>> BulletImpactInteraction('impact', target='pendulum').target
        'pendulum'
    """

    law_classes = (RateLaw, EventLaw)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the free flight of the bullet and the event of the impact.

        Args:
            context:
                The namespace of the bullet and pendulum and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, p = context.t, context.target
            velocity = p.x.diff(t)
            reset = {velocity: 0, p.theta.diff(t): p.m * velocity / ((p.m + p.M) * p.l)}
            self.laws_ = [RateLaw(p.x.diff(t, 2), 0), EventLaw(p.x, reset, direction=1)]


class BallisticPendulumSystem(systems.BaseSystem):
    """A bullet that sticks in the block of a pendulum at rest.

    The pendulum then swings up.

    The bullet flies horizontally towards the block, and the impact is an event. Momentum is conserved in the perfectly
    inelastic collision, so the pendulum starts with the speed `m v / (m + M)` and rises to the height `(m v / (m + M))²
    / (2 g)`. The bullet position stops at the block after the impact. The system has the bullet position `pendulum.x`
    and the angle `pendulum.theta`, and their velocities `pendulum.x'` and `pendulum.theta'`.

    Args:
        speed:
            The speed of the bullet.

        distance:
            The initial distance of the bullet from the block.

        m:
            The mass of the bullet.

        M:
            The mass of the block.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BallisticPendulumSystem
        >>> BallisticPendulumSystem(distance=2.0).prepare().system_.elements[0].x
        -2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'events', 'impacts', 'momentum', 'hybrid')
    references = ('Ryzhov et al. 2021',)
    until = 3.0
    dt = 0.01

    def __init__(
        self,
        speed: float = 200.0,
        distance: float = 1.0,
        *,
        m: float = 0.01,
        M: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.speed = speed
        self.distance = distance
        self.m = m
        self.M = M
        self.l = l
        self.g = g

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a bullet that sticks in the block of a pendulum at rest.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        pendulum = BallisticPendulumElement(
            'pendulum',
            x=-self.distance,
            m=self.m,
            M=self.M,
            l=self.l,
            g=self.g,
            values={"x'": self.speed},
        )
        self.system_ = systems.System(
            self.label,
            elements=[pendulum],
            interactions=[
                BlockSwingInteraction('swing', target='pendulum'),
                BulletImpactInteraction('impact', target='pendulum'),
            ],
        )


class HuygensClocksSystem(BaseSystem):
    """Pendulum clocks on supports joined by springs.

    The clocks synchronize through the motion of the supports.

    Every pendulum hangs from a support of mass `M` that moves horizontally with damping `rho`, and neighbouring
    supports are joined by springs of stiffness `k`. A pendulum follows `m l θ'' + f(θ) θ' + m g sin θ = -m x'' cos θ`
    and its support `(M + m) x'' + 2 rho x' + m l θ'' cos θ = m l θ'² sin θ + spring forces`. The escapement `f(θ) = 2 λ
    sgn(|θ| - θ_c)` is `-2 λ` for `|θ| < θ_c`, pumping energy in, and `2 λ` beyond, so every clock has a limit cycle.
    With `λ = 0` the clocks are plain pendulums. Two identical clocks synchronize in antiphase. The first clock starts
    at `theta`, away from the switch of the escapement at `θ_c`, and the others at rest. The system has the angles
    `clock_<i>.theta` and the support positions `clock_<i>.x`, all of order 2.

    Args:
        n_clocks:
            The number of clocks in the row.

        theta:
            The initial angle of the first clock. The others start at rest.

        k:
            The stiffness of the springs between neighbouring supports.

        m:
            The mass of every bob.

        l:
            The length of every pendulum.

        M:
            The mass of every support.

        g:
            The gravitational acceleration.

        rho:
            The damping of the supports.

        lam:
            The strength of the escapement.

        theta_c:
            The angle below which the escapement pumps energy in.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import HuygensClocksSystem
        >>> len(HuygensClocksSystem(3).prepare().system_.interactions)
        5
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'synchronization', 'limit cycle')
    references = ('Huygens 1673', 'Dilão 2023', 'Dilão 2023')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        n_clocks: int = 2,
        *,
        theta: float = 0.2,
        k: float = 1.0,
        m: float = 1.0,
        l: float = 1.0,
        M: float = 1.0,
        g: float = 9.8,
        rho: float = 0.1,
        lam: float = 0.1,
        theta_c: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_clocks = n_clocks
        self.theta = theta
        self.k = k
        self.m = m
        self.l = l
        self.M = M
        self.g = g
        self.rho = rho
        self.lam = lam
        self.theta_c = theta_c

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        clocks = [
            ClockPendulumElement(
                f'clock_{index}',
                theta=self.theta if index == 0 else 0.0,
                m=self.m,
                l=self.l,
                M=self.M,
                g=self.g,
                rho=self.rho,
                lam=self.lam,
                theta_c=self.theta_c,
            )
            for index in range(self.n_clocks)
        ]
        springs = [
            SupportSpringInteraction(f'spring_{index}', first=f'clock_{index}', second=f'clock_{index + 1}', k=self.k)
            for index in range(self.n_clocks - 1)
        ]
        motions = [
            ClockPendulumInteraction(f'motion_{index}', target=f'clock_{index}') for index in range(self.n_clocks)
        ]
        return CompositeSystem(self.label, elements=clocks, interactions=[*springs, *motions])
