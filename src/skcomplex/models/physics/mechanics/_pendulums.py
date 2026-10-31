"""Pendulum models: double, coupled, parametric, magnetic, and ballistic pendulums, carts, and clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sign, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import CrossingInteraction, EulerLagrangeInteraction, Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.particles import (
    DampingInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
    PotentialEnergyInteraction,
)
from ._mechanics import PendulumElement, PendulumInteraction


def _double_bobs(theta1: float, theta2: float, l1: float, l2: float, **_: float) -> np.ndarray:
    """Return the positions of the two bobs of a double pendulum hanging from the origin."""
    first = np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)])
    return np.stack([first, first + np.array([l2 * np.sin(theta2), -l2 * np.cos(theta2)])])


def _cart_and_bob(y: float, theta: float, l: float, **_: float) -> np.ndarray:
    """Return the positions of a cart on a horizontal track and of the bob that hangs from it."""
    return np.array([[y, 0.0], [y + l * np.sin(theta), -l * np.cos(theta)]])


class DoublePendulumElement(Element):
    """Double pendulum, a bob of mass `m2` on a rod of length `l2` hanging from a bob of mass `m1` on a rod `l1`.

    The angles `theta1` and `theta2` of the rods from the vertical have order 2. The motion comes from
    `DoublePendulumInteraction`.

    Args:
        label:
            The label of the double pendulum.

        visual:
            How the double pendulum is drawn, or `None` for its class visual.

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
        >>> from skcomplex.models.physics.mechanics import DoublePendulumElement
        >>> DoublePendulumElement('pendulum', m2=2.0).m2
        2.0
    """

    visual = VisualSpec('bodies', {'positions': _double_bobs, 'pivot': (0.0, 0.0), 'rods': True, 'traced': [1]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
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
        super().__init__(label, visual=visual)
        self.theta1 = theta1
        self.theta1_d1 = theta1_d1
        self.theta2 = theta2
        self.theta2_d1 = theta2_d1
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g


class DoublePendulumInteraction(EulerLagrangeInteraction):
    """Motion of a double pendulum, the Euler-Lagrange equations of its kinetic energy and gravity.

    The Lagrangian is `(m1 + m2) l1² θ1'² / 2 + m2 l2² θ2'² / 2 + m2 l1 l2 θ1' θ2' cos(θ1 - θ2) + (m1 + m2) g l1 cos θ1
    + m2 g l2 cos θ2`, with the masses, lengths, and gravity of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the double pendulum.

    Examples:
        >>> from skcomplex.models.physics.mechanics import DoublePendulumInteraction
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


class PendulumSpringInteraction(Interaction):
    """Spring between the bobs of two pendulums, which stretches with the horizontal distance of the bobs.

    The spring of stiffness `k` has the energy `k (l2 sin θ2 - l1 sin θ1)² / 2`, whose torque on every pendulum,
    divided by its moment of inertia `m l²`, adds to its angular acceleration.

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
        >>> from skcomplex.models.physics.mechanics import PendulumSpringInteraction
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
        super().__init__(label)
        self.first = first
        self.second = second
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the angular accelerations the spring gives the two pendulums.

        Args:
            s:
                The namespace of the symbols the laws can use, with `s.first` and `s.second`.

        Returns:
            laws:
                The rate laws of the two angles.
        """
        stretch = s.second.l * sin(s.second.theta) - s.first.l * sin(s.first.theta)
        return [
            RateLaw(s.first.theta, s.k * stretch * cos(s.first.theta) / (s.first.m * s.first.l)),
            RateLaw(s.second.theta, -s.k * stretch * cos(s.second.theta) / (s.second.m * s.second.l)),
        ]


class CartPendulumElement(Element):
    """Cart on a horizontal track at the position `y`, with a pendulum of angle `theta` hanging from it.

    The position and the angle have order 2. The motion comes from `CartPendulumInteraction`.

    Args:
        label:
            The label of the cart and pendulum.

        visual:
            How the cart and pendulum are drawn, or `None` for its class visual.

        y:
            The initial position of the cart.

        y_d1:
            The initial velocity of the cart.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

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

    Examples:
        >>> from skcomplex.models.physics.mechanics import CartPendulumElement
        >>> CartPendulumElement('cart', m1=5.0).m1
        5.0
    """

    visual = VisualSpec('bodies', {'positions': _cart_and_bob, 'rods': True, 'traced': [1]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec(order=2)] = 0.0,
        y_d1: float = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        F: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.y_d1 = y_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.F = F


class CartPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a cart and the pendulum that hangs from it, the Euler-Lagrange equations with a force on the cart.

    The Lagrangian is `(m1 + m2) y'² / 2 + m2 l y' θ' cos θ + m2 l² θ'² / 2 + m2 g l cos θ`, and the force `F` of the
    target drives the position `y` of the cart.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the cart and pendulum.

    Examples:
        >>> from skcomplex.models.physics.mechanics import CartPendulumInteraction
        >>> CartPendulumInteraction('motion', target='cart').states
        ('y', 'theta')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('y', 'theta'), forcing=self._get_forces_on_coordinates)

    @staticmethod
    def _get_forces_on_coordinates(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the force `F` on the cart and none on the angle."""
        return [namespace.target.F, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the cart and pendulum.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        v, w = derivative(p.y, 1), derivative(p.theta, 1)
        kinetic = (p.m1 + p.m2) * v**2 / 2 + p.m2 * p.l * v * w * cos(p.theta) + p.m2 * p.l**2 * w**2 / 2
        return kinetic + p.m2 * p.g * p.l * cos(p.theta)


class ParametricPendulumElement(Element):
    """Pendulum whose suspension point oscillates vertically, with the angle `theta` of order 2.

    The motion comes from `ParametricPendulumInteraction`.

    Args:
        label:
            The label of the pendulum.

        visual:
            How the pendulum is drawn, or `None` for none.

        theta:
            The initial angle.

        theta_d1:
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

    Examples:
        >>> from skcomplex.models.physics.mechanics import ParametricPendulumElement
        >>> ParametricPendulumElement('swing', epsilon=0.5).epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.2,
        Omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2 * math.sqrt(9.81),
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l
        self.epsilon = epsilon
        self.Omega = Omega
        self.c = c


class ParametricPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a parametric pendulum, the Euler-Lagrange equation of `θ'² / 2 + (g / l)(1 + ε cos Ω t) cos θ`.

    The Rayleigh dissipation `c θ'² / 2` adds the friction, so the angle follows
    `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0`, with the parameters of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pendulum.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ParametricPendulumInteraction
        >>> ParametricPendulumInteraction('swing', target='pendulum').states
        'theta'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states='theta', dissipation=self._get_friction)

    @staticmethod
    def _get_friction(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation `c θ'² / 2` of the friction."""
        return namespace.target.c * derivative(namespace.target.theta, 1) ** 2 / 2

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the parametric pendulum per moment of inertia.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        gravity = p.g / p.l * (1 + p.epsilon * cos(p.Omega * namespace.t))
        return derivative(p.theta, 1) ** 2 / 2 + gravity * cos(p.theta)


class BallisticPendulumElement(Element):
    """Bullet of mass `m` at the position `x` flying towards the block of mass `M` of a pendulum with the angle `theta`.

    The position and the angle have order 2. The bullet flies freely, the pendulum swings, and the impact transfers
    the momentum of the bullet to the pendulum.

    Args:
        label:
            The label of the bullet and pendulum.

        visual:
            How the bullet and pendulum are drawn, or `None` for none.

        x:
            The initial position of the bullet, negative before the block.

        x_d1:
            The initial velocity of the bullet.

        theta:
            The initial angle of the pendulum.

        theta_d1:
            The initial angular velocity of the pendulum.

        m:
            The mass of the bullet.

        M:
            The mass of the block.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.models.physics.mechanics import BallisticPendulumElement
        >>> BallisticPendulumElement('pendulum', M=2.0).M
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = -1.0,
        x_d1: float = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        theta_d1: float = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.M = M
        self.l = l
        self.g = g


class ClockPendulumElement(Element):
    """Pendulum clock, a pendulum of angle `theta` hanging from a support at the horizontal position `x`.

    The angle and the position have order 2. The motion comes from `ClockPendulumInteraction`.

    Args:
        label:
            The label of the clock.

        visual:
            How the clock is drawn, or `None` for none.

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
        >>> from skcomplex.models.physics.mechanics import ClockPendulumElement
        >>> ClockPendulumElement('clock', lam=0.2).lam
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
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
        super().__init__(label, visual=visual)
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


class ClockPendulumInteraction(EulerLagrangeInteraction):
    """Motion of a pendulum clock on a moving support, the Euler-Lagrange equations with the escapement.

    The Lagrangian is `(M + m) x'² / 2 + m l x' θ' cos θ + m l² θ'² / 2 + m g l cos θ`. The Rayleigh dissipation
    `λ l sgn(|θ| - θ_c) θ'² + rho x'²` damps the support and gives the escapement `2 λ l sgn(|θ| - θ_c) θ'`, which pumps
    energy into the pendulum below the angle `θ_c` and takes it out beyond.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the clock.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ClockPendulumInteraction
        >>> ClockPendulumInteraction('motion', target='clock').states
        ('theta', 'x')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('theta', 'x'), dissipation=self._get_escapement)

    @staticmethod
    def _get_escapement(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation of the escapement and of the damping of the support."""
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


class SupportSpringInteraction(Interaction):
    """Spring of stiffness `k` between the supports of two pendulum clocks.

    The spring pulls the supports by the force `k (x2 - x1)`, and every clock answers with the accelerations of its
    support and its pendulum, through the inverse of its mass matrix.

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
        >>> from skcomplex.models.physics.mechanics import SupportSpringInteraction
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
        super().__init__(label)
        self.first = first
        self.second = second
        self.k = k

    def _get_response(self, clock: SimpleNamespace, force: Expression) -> list[BaseLaw]:
        """Return the accelerations of a clock's support and pendulum under a horizontal force on the support."""
        inertia = clock.M + clock.m * sin(clock.theta) ** 2
        return [
            RateLaw(clock.x, force / inertia),
            RateLaw(clock.theta, -cos(clock.theta) * force / (clock.l * inertia)),
        ]

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations the spring gives the supports and pendulums of both clocks.

        Args:
            s:
                The namespace of the symbols the laws can use, with `s.first` and `s.second`.

        Returns:
            laws:
                The rate laws of the positions and angles of both clocks.
        """
        force = s.k * (s.second.x - s.first.x)
        return [*self._get_response(s.first, force), *self._get_response(s.second, -force)]


class MagnetInteraction(PotentialEnergyInteraction):
    """Magnet below the plane of a pendulum bob, which attracts it with energy `-gamma / sqrt(|x - center|**2 + d**2)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bob.

        center:
            The position of the magnet in the plane.

        gamma:
            The strength of the magnet.

        d:
            The height of the bob above the plane of the magnets.

    Examples:
        >>> from skcomplex.models.physics.mechanics import MagnetInteraction
        >>> MagnetInteraction('magnet', target='bob', center=[1.0, 0.0]).gamma
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, target=target)
        self.center = center
        self.gamma = gamma
        self.d = d

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `-gamma / sqrt(|x - center|**2 + d**2)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        displacement = s.target.x - s.center
        return -s.gamma / sqrt(displacement.dot_product(displacement) + s.d**2)


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'chaos', 'lagrangian'),
    references=('Ryzhov et al. 2021', 'Argun et al. 2021'),
    defaults={'until': 20.0, 'dt': 0.02},
)
def double_pendulum(
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
) -> System:
    """Return a double pendulum, a pendulum hanging from the bob of another, written as its two Lagrange equations.

    The equations `(m1 + m2) l1 θ1'' + m2 l2 θ2'' cos(θ1 - θ2) + m2 l2 θ2'² sin(θ1 - θ2) + (m1 + m2) g sin θ1 = 0`
    and `l2 θ2'' + l1 θ1'' cos(θ1 - θ2) - l1 θ1'² sin(θ1 - θ2) + g sin θ2 = 0` are solved jointly for the angular
    accelerations. The motion is chaotic at large amplitudes.

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

    Returns:
        system:
            The model, with the angles `pendulum.theta1` and `pendulum.theta2` of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import double_pendulum
        >>> double_pendulum(m2=2.0).elements[0].m2
        2.0
    """
    return System(
        elements=[
            DoublePendulumElement(
                'pendulum',
                theta1=theta1,
                theta2=theta2,
                theta1_d1=omega1,
                theta2_d1=omega2,
                m1=m1,
                m2=m2,
                l1=l1,
                l2=l2,
                g=g,
            ),
        ],
        interactions=[DoublePendulumInteraction('motion', target='pendulum')],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'oscillation', 'normal modes'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 40.0, 'dt': 0.05},
)
def coupled_pendulums(
    theta1: float = 0.2,
    theta2: float = 0.0,
    *,
    k: float = 0.5,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return two pendulums whose bobs are joined by a spring, which pass their energy back and forth.

    The spring stretches with the horizontal distance `l2 sin θ2 - l1 sin θ1` of the bobs and adds the torque
    `k l1 (l2 sin θ2 - l1 sin θ1) cos θ1` to the first pendulum and the opposite one to the second. Identical pendulums
    have the normal modes `sqrt(g / l)` in phase and `sqrt(g / l + 2 k / m)` in antiphase.

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

    Returns:
        system:
            The model, with the pendulums `left` and `right` and their angles `left.theta` and `right.theta`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import coupled_pendulums
        >>> coupled_pendulums(k=2.0).interactions[0].k
        2.0
    """
    return System(
        elements=[
            PendulumElement('left', theta=theta1, m=m1, l=l1, g=g),
            PendulumElement('right', theta=theta2, m=m2, l=l2, g=g),
        ],
        interactions=[
            PendulumSpringInteraction('spring', first='left', second='right', k=k),
            PendulumInteraction('left_swing', target='left'),
            PendulumInteraction('right_swing', target='right'),
        ],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'lagrangian', 'momentum'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 10.0, 'dt': 0.02},
)
def cart_pendulum(
    theta: float = 0.5,
    y: float = 0.0,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
    F: float = 0.0,
) -> System:
    """Return a pendulum hanging from a cart that slides without friction along a horizontal track.

    The Lagrange equations `(m1 + m2) y'' + m2 l (θ'' cos θ - θ'² sin θ) = F` and `l θ'' + y'' cos θ + g sin θ = 0`
    couple the position `y` of the cart and the angle `θ` of the pendulum. Without a force the horizontal momentum is
    conserved, so the cart swings against the pendulum.

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

    Returns:
        system:
            The model, with the cart position `cart.y` and the angle `cart.theta`, both of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import cart_pendulum
        >>> cart_pendulum(m1=5.0).elements[0].m1
        5.0
    """
    return System(
        elements=[CartPendulumElement('cart', y=y, theta=theta, m1=m1, m2=m2, l=l, g=g, F=F)],
        interactions=[CartPendulumInteraction('motion', target='cart')],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'forcing', 'parametric resonance'),
    references=('Dilão 2023',),
    defaults={'until': 30.0, 'dt': 0.02},
)
def parametric_pendulum(
    theta: float = 0.1,
    omega: float = 0.0,
    *,
    g: float = 9.81,
    l: float = 1.0,
    epsilon: float = 0.2,
    Omega: float = 2 * math.sqrt(9.81),
    c: float = 0.0,
) -> System:
    """Return a pendulum whose suspension point oscillates vertically, `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0`.

    The vertical acceleration of the suspension point modulates gravity. Small oscillations grow when `Ω` is close to
    twice the natural frequency `sqrt(g / l)`, the parametric resonance of a swing.

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

    Returns:
        system:
            The model, with the angle `swing.theta` of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import parametric_pendulum
        >>> parametric_pendulum(epsilon=0.5).elements[0].epsilon
        0.5
    """
    return System(
        elements=[
            ParametricPendulumElement(
                'swing',
                theta=theta,
                theta_d1=omega,
                g=g,
                l=l,
                epsilon=epsilon,
                Omega=Omega,
                c=c,
            ),
        ],
        interactions=[ParametricPendulumInteraction('motion', target='swing')],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'multistability', 'fractal basins'),
    references=('Datseris and Parlitz 2022',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def magnetic_pendulum(
    x: float = 1.0,
    y: float = 1.5,
    *,
    n_magnets: int = 3,
    omega: float = 1.0,
    q: float = 0.2,
    gamma: float = 1.0,
    d: float = 0.3,
) -> System:
    """Return a magnetic bob on a string swinging above magnets on a circle, which comes to rest at one of them.

    The bob follows `x'' = -ω² x - q x' - Σ gamma (x - x_i) / d_i³` with `d_i = sqrt(|x - x_i|² + d²)`. The basins of
    the magnets have fractal boundaries.

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

    Returns:
        system:
            The model, with the bob `bob` and its position `bob.x` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.mechanics import magnetic_pendulum
        >>> len(magnetic_pendulum(n_magnets=4).interactions)
        6
    """
    angles = 2 * np.pi * np.arange(n_magnets) / n_magnets
    magnets = [
        MagnetInteraction(
            f'magnet_{index}',
            target='bob',
            center=[round(float(np.cos(angle)), 12), round(float(np.sin(angle)), 12)],
            gamma=gamma,
            d=d,
        )
        for index, angle in enumerate(angles)
    ]
    return System(
        elements=[MassPointElement('bob', x=[x, y], x_d1=[0.0, 0.0])],
        interactions=[
            HarmonicPotentialInteraction('string', target='bob', k=omega**2),
            DampingInteraction('friction', target='bob', gamma=q),
            *magnets,
        ],
        space=ContinuousSpace(n_dim=2),
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'events', 'impacts', 'momentum', 'hybrid'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 3.0, 'dt': 0.01},
)
def ballistic_pendulum(
    speed: float = 200.0,
    distance: float = 1.0,
    *,
    m: float = 0.01,
    M: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return a bullet that sticks in the block of a pendulum at rest, which then swings up.

    The bullet flies horizontally towards the block, and the impact is an event. Momentum is conserved in the
    perfectly inelastic collision, so the pendulum starts with the speed `m v / (m + M)` and rises to the height
    `(m v / (m + M))² / (2 g)`. The bullet position stops at the block after the impact.

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

    Returns:
        system:
            The model, with the bullet position `pendulum.x` and the angle `pendulum.theta`, both of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ballistic_pendulum
        >>> ballistic_pendulum(distance=2.0).elements[0].x
        -2.0
    """
    swing = EulerLagrangeInteraction(
        'swing',
        target='pendulum',
        states='theta',
        lagrangian=lambda s: s.target.l * derivative(s.target.theta, 1) ** 2 / 2 + s.target.g * cos(s.target.theta),
    )
    impact = CrossingInteraction(
        'impact',
        target='pendulum',
        guard=lambda s: s.target.x,
        reset={
            'x_d1': lambda s: 0,
            'theta_d1': lambda s: s.target.m * s.target.x_d1 / ((s.target.m + s.target.M) * s.target.l),
        },
        direction=1,
    )
    return System(
        elements=[BallisticPendulumElement('pendulum', x=-distance, x_d1=speed, m=m, M=M, l=l, g=g)],
        interactions=[swing, impact],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'synchronization', 'limit cycle'),
    references=('Huygens 1673', 'Dilão 2023', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def huygens_clocks(
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
) -> System:
    """Return pendulum clocks on supports joined by springs, which synchronize through the motion of the supports.

    Every pendulum hangs from a support of mass `M` that moves horizontally with damping `rho`, and neighbouring
    supports are joined by springs of stiffness `k`. A pendulum follows `m l θ'' + f(θ) θ' + m g sin θ = -m x'' cos θ`
    and its support `(M + m) x'' + 2 rho x' + m l θ'' cos θ = m l θ'² sin θ + spring forces`. The escapement
    `f(θ) = 2 λ sgn(|θ| - θ_c)` is `-2 λ` for `|θ| < θ_c`, pumping energy in, and `2 λ` beyond, so every clock has a
    limit cycle. With `λ = 0` the clocks are plain pendulums. Two identical clocks synchronize in antiphase. The
    first clock starts at `theta`, away from the switch of the escapement at `θ_c`, and the others at rest.

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

    Returns:
        system:
            The model, with the angles `clock_<i>.theta` and the support positions `clock_<i>.x`, all of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import huygens_clocks
        >>> len(huygens_clocks(3).interactions)
        5
    """
    clocks = [
        ClockPendulumElement(
            f'clock_{index}',
            theta=theta if index == 0 else 0.0,
            m=m,
            l=l,
            M=M,
            g=g,
            rho=rho,
            lam=lam,
            theta_c=theta_c,
        )
        for index in range(n_clocks)
    ]
    springs = [
        SupportSpringInteraction(f'spring_{index}', first=f'clock_{index}', second=f'clock_{index + 1}', k=k)
        for index in range(n_clocks - 1)
    ]
    motions = [ClockPendulumInteraction(f'motion_{index}', target=f'clock_{index}') for index in range(n_clocks)]
    return System(elements=clocks, interactions=[*springs, *motions])
