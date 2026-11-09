"""Celestial mechanics systems of spin-orbit coupling and relativistic orbits.

The module holds the spin-orbit coupling of a dumbbell and relativistic orbits.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class DumbbellElement(Element):
    """Rigid dumbbell orbiting a primary.

    The element holds the position of its centre of mass and the orientation of its arm.

    The cylindrical coordinates `(u, theta, v)` place the centre of mass of the dumbbell relative to the primary, and
    the spherical angles `(phi, psi)` give the direction of its arm, of length `epsilon` in units of the mean orbit
    radius. The mass `1 - delta` of the dumbbell sits at `delta epsilon` along the arm and the mass `delta` at
    `1 - delta` against it. All five coordinates are states of order 2 that a Lagrangian moves.

    Args:
        label:
            The label of the dumbbell.

        visual:
            How the dumbbell is drawn, or `None` for `default_visual_`.

        u:
            The initial distance from the axis through the primary.

        u_d1:
            The initial radial velocity.

        theta:
            The initial orbital angle.

        theta_d1:
            The initial orbital angular velocity.

        v:
            The initial height above the orbital plane.

        v_d1:
            The initial vertical velocity.

        phi:
            The initial azimuth of the arm.

        phi_d1:
            The initial spin of the arm about the vertical.

        psi:
            The initial polar angle of the arm, `pi / 2` in the orbital plane.

        psi_d1:
            The initial rate of the polar angle of the arm.

        delta:
            The share of the second mass in the mass of the dumbbell, `1 / 2` for equal masses.

        epsilon:
            The length of the arm in units of the mean orbit radius.

        mu:
            The share of the primary in the mass of the three bodies.

    Attributes:
        default_visual_:
            The visual of the dumbbell when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DumbbellElement
        >>> DumbbellElement('dumbbell').delta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(order=2)] = 1.0,
        u_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.0,
        theta_d1: float = 1.0,
        v: Annotated[float, StateSpec(order=2)] = 0.0,
        v_d1: float = 0.0,
        phi: Annotated[float, StateSpec(order=2)] = 0.0,
        phi_d1: float = 1.0,
        psi: Annotated[float, StateSpec(order=2)] = math.pi / 2,
        psi_d1: float = 0.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.5,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.u_d1 = u_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.v = v
        self.v_d1 = v_d1
        self.phi = phi
        self.phi_d1 = phi_d1
        self.psi = psi
        self.psi_d1 = psi_d1
        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu


COORDINATES = ('u', 'theta', 'v', 'phi', 'psi')


def _get_dumbbell_distances(d: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the distances `rho12`, `rho13` from the primary to both dumbbell masses."""
    radius = d.u**2 + d.v**2
    projection = d.u * sin(d.psi) * cos(d.theta - d.phi) + d.v * cos(d.psi)
    first = radius + d.delta**2 * d.epsilon**2 + 2 * d.delta * d.epsilon * projection
    second = radius + (1 - d.delta) ** 2 * d.epsilon**2 - 2 * (1 - d.delta) * d.epsilon * projection
    return sqrt(first), sqrt(second)


def _get_dumbbell_parts(d: SimpleNamespace, rates: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the kinetic and potential energies of a dumbbell per reduced orbit mass.

    The energies follow from given rates.
    """
    inertia = d.delta * (1 - d.delta) / d.mu * d.epsilon**2
    orbit = rates.u**2 + d.u**2 * rates.theta**2 + rates.v**2
    spin = inertia * (rates.psi**2 + sin(d.psi) ** 2 * rates.phi**2)
    first, second = _get_dumbbell_distances(d)
    return (orbit + spin) / 2, -(1 - d.delta) / first - d.delta / second


def _get_dumbbell_energy(d: SimpleNamespace) -> Expression:
    """Return the energy of a dumbbell, the sum of its kinetic and potential energies."""
    rates = SimpleNamespace(**{name: getattr(d, f'{name}_d1') for name in COORDINATES})
    kinetic, potential = _get_dumbbell_parts(d, rates)
    return kinetic + potential


def _get_dumbbell_momentum(d: SimpleNamespace) -> Expression:
    """Return the angular momentum of a dumbbell about the normal of the orbital plane."""
    inertia = d.delta * (1 - d.delta) / d.mu * d.epsilon**2
    return d.u**2 * d.theta_d1 + inertia * sin(d.psi) ** 2 * d.phi_d1


class KeplerianDumbbellSystem(BaseSystem):
    """The Keplerian dumbbell.

    Two masses on a rigid massless arm orbit a primary under gravity.

    The centre of mass of the dumbbell at the cylindrical coordinates `(u, theta, v)` about the primary and the arm at
    the spherical angles `(phi, psi)` move under the dimensionless Lagrangian `L = (u'² + u² theta'² + v'²) / 2 + I
    (psi'² + sin² psi phi'²) / 2 + (1 - delta) / rho12 + delta / rho13`, with the moment of inertia `I = delta (1 -
    delta) epsilon² / mu` of the arm. The distances of the two masses from the primary are `rho12² = u² + v² + delta²
    epsilon² + 2 delta epsilon P` and `rho13² = u² + v² + (1 - delta)² epsilon² - 2 (1 - delta) epsilon P`, with `P = u
    sin psi cos(theta - phi) + v cos psi`. Its Euler-Lagrange equations are the equations of motion of Dilão and
    Murteira. The gravity gradient across the arm couples the spin to the orbit, the simplest spin-orbit coupling of a
    rigid body. For `epsilon = 0` the orbit is a Kepler orbit, of period `2π` at `u = 1`. The energy and the angular
    momentum about the normal to the orbital plane are conserved. By default the arm spins three times faster than it
    orbits, in the orbital plane, and its spin and orbit exchange angular momentum. The system has the coordinates
    `dumbbell.u`, `dumbbell.theta`, `dumbbell.v`, `dumbbell.phi`, and `dumbbell.psi` of order 2, and the observables
    `dumbbell.energy` and `dumbbell.angular_momentum`.

    Args:
        u:
            The initial distance of the centre of mass from the axis through the primary.

        vu:
            The initial radial velocity.

        theta:
            The initial orbital angle.

        omega:
            The initial orbital angular velocity.

        phi:
            The initial azimuth of the arm.

        spin:
            The initial spin of the arm about the normal to the orbital plane.

        v:
            The initial height above the orbital plane.

        vv:
            The initial vertical velocity.

        psi:
            The initial polar angle of the arm, `pi / 2` in the orbital plane.

        vpsi:
            The initial rate of the polar angle of the arm.

        delta:
            The share of the second mass in the mass of the dumbbell, `1 / 2` for equal masses.

        epsilon:
            The length of the arm in units of the mean orbit radius.

        mu:
            The share of the primary in the mass of the three bodies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import KeplerianDumbbellSystem
        >>> KeplerianDumbbellSystem(epsilon=0.2).prepare().system_.elements[0].epsilon
        0.2
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'second order', 'gravity', 'spin-orbit coupling', 'lagrangian')
    references = ('Dilão and Murteira 2020', 'Dilão 2023')
    until = 60.0
    dt = 0.05
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
    invariants: ClassVar[Mapping[str, float]] = {'dumbbell.energy': 1e-6, 'dumbbell.angular_momentum': 1e-6}
    observables = (
        ObservableSpec('energy', _get_dumbbell_energy, element='dumbbell'),
        ObservableSpec('angular_momentum', _get_dumbbell_momentum, element='dumbbell'),
    )

    def __init__(
        self,
        u: float = 1.0,
        vu: float = 0.0,
        theta: float = 0.0,
        omega: float = 1.0,
        *,
        phi: float = 0.0,
        spin: float = 3.0,
        v: float = 0.0,
        vv: float = 0.0,
        psi: float = 1.5707963267948966,
        vpsi: float = 0.0,
        delta: float = 0.5,
        epsilon: float = 0.1,
        mu: float = 0.9,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.u = u
        self.vu = vu
        self.theta = theta
        self.omega = omega
        self.phi = phi
        self.spin = spin
        self.v = v
        self.vv = vv
        self.psi = psi
        self.vpsi = vpsi
        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu

    @staticmethod
    def _get_dumbbell_lagrangian(s: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the Keplerian dumbbell of the target."""
        d = s.target
        rates = SimpleNamespace(**{name: derivative(getattr(d, name), 1) for name in COORDINATES})
        kinetic, potential = _get_dumbbell_parts(d, rates)
        return kinetic - potential

    def _build_system(self) -> CompositeSystem:
        """Return the Keplerian dumbbell.

        Two masses on a rigid massless arm orbit a primary under gravity.
        """
        dumbbell = DumbbellElement(
            'dumbbell',
            u=self.u,
            u_d1=self.vu,
            theta=self.theta,
            theta_d1=self.omega,
            v=self.v,
            v_d1=self.vv,
            phi=self.phi,
            phi_d1=self.spin,
            psi=self.psi,
            psi_d1=self.vpsi,
            delta=self.delta,
            epsilon=self.epsilon,
            mu=self.mu,
        )
        motion = EulerLagrangeInteraction(
            'motion',
            target='dumbbell',
            states=COORDINATES,
            lagrangian=self._get_dumbbell_lagrangian,
        )
        return CompositeSystem(self.label, elements=[dumbbell], interactions=[motion])


class RelativisticOrbitSystem(DynamicalSystem):
    """Einstein's orbit equation `u'' + u = alpha + epsilon u²` of a planet.

    The perihelion of the planet precesses.

    The inverse radius `u = 1 / r` of the planet is a function of its polar angle, which the simulation time stands for,
    so the orbit in the plane is `(cos t / u, sin t / u)`. For `epsilon = 0` the orbit is a closed Newtonian ellipse of
    eccentricity `|u(0) / alpha - 1|` for `u'(0) = 0`. The term `epsilon u²` is the first relativistic correction, and
    with it the orbit is a rosette whose perihelion advances by about `2π epsilon alpha` every revolution. The equation
    has two fixed points, the roots `(1 ∓ sqrt(1 - 4 alpha epsilon)) / (2 epsilon)` of `epsilon u² - u + alpha`. The
    smaller one, close to `alpha`, is a centre, the stable circular orbit, and the larger one is a saddle, the unstable
    circular orbit close to the black hole, beyond which a planet falls in. The system has the inverse radius `planet.u`
    of order 2 along the angle and its derivative `planet.u_d1`.

    Args:
        u:
            The initial inverse radius.

        du:
            The initial derivative of the inverse radius along the angle.

        alpha:
            The Newtonian inverse semi-latus rectum.

        epsilon:
            The strength of the relativistic correction.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RelativisticOrbitSystem
        >>> RelativisticOrbitSystem(epsilon=0.0).prepare().system_.elements[0].get_value('epsilon')
        0.0
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'second order', 'gravity', 'general relativity', 'precession')
    references = ('Einstein 1915', 'Dilão 2023')
    until = 60.0
    dt = 0.02

    def __init__(
        self,
        u: float = 1.5,
        du: float = 0.0,
        *,
        alpha: float = 1.0,
        epsilon: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.u = u
        self.du = du
        self.alpha = alpha
        self.epsilon = epsilon

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'planet'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'u': DeclaredVariableSpec(self.u, StateSpec(order=2, initial=(self.du,))),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec(domain=RealSet.open(0, oo))),
            'epsilon': DeclaredVariableSpec(self.epsilon, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return Einstein's orbit equation, the second derivative of `u` along the angle."""
        return {'u': lambda s: s.target.alpha + s.target.epsilon * s.target.u**2 - s.target.u}
