"""Celestial mechanics systems: spin-orbit coupling of a dumbbell and relativistic orbits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, cos, sin, sqrt

from ....core.components.interactions import EulerLagrangeInteraction
from ....core.components.laws.utils import derivative
from ....core.components.system import System
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._dumbbell import DumbbellElement
from ._relativistic_orbit import RelativisticOrbitElement

COORDINATES = ('u', 'theta', 'v', 'phi', 'psi')


def _get_dumbbell_distances(d: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the distances `rho12` and `rho13` from the primary to the two masses of a dumbbell."""
    radius = d.u**2 + d.v**2
    projection = d.u * sin(d.psi) * cos(d.theta - d.phi) + d.v * cos(d.psi)
    first = radius + d.delta**2 * d.epsilon**2 + 2 * d.delta * d.epsilon * projection
    second = radius + (1 - d.delta) ** 2 * d.epsilon**2 - 2 * (1 - d.delta) * d.epsilon * projection
    return sqrt(first), sqrt(second)


def _get_dumbbell_parts(d: SimpleNamespace, rates: SimpleNamespace) -> tuple[Expression, Expression]:
    """Return the kinetic and potential energies of a dumbbell per reduced mass of its orbit, from given rates."""
    inertia = d.delta * (1 - d.delta) / d.mu * d.epsilon**2
    orbit = rates.u**2 + d.u**2 * rates.theta**2 + rates.v**2
    spin = inertia * (rates.psi**2 + sin(d.psi) ** 2 * rates.phi**2)
    first, second = _get_dumbbell_distances(d)
    return (orbit + spin) / 2, -(1 - d.delta) / first - d.delta / second


def _get_dumbbell_lagrangian(s: SimpleNamespace) -> Expression:
    """Return the Lagrangian of the Keplerian dumbbell of the target."""
    d = s.target
    rates = SimpleNamespace(**{name: derivative(getattr(d, name), 1) for name in COORDINATES})
    kinetic, potential = _get_dumbbell_parts(d, rates)
    return kinetic - potential


def _get_dumbbell_energy(d: SimpleNamespace) -> Expression:
    """Return the energy of a dumbbell, the sum of its kinetic and potential energies."""
    rates = SimpleNamespace(**{name: getattr(d, f'{name}_d1') for name in COORDINATES})
    kinetic, potential = _get_dumbbell_parts(d, rates)
    return kinetic + potential


def _get_dumbbell_momentum(d: SimpleNamespace) -> Expression:
    """Return the angular momentum of a dumbbell about the axis normal to the orbital plane."""
    inertia = d.delta * (1 - d.delta) / d.mu * d.epsilon**2
    return d.u**2 * d.theta_d1 + inertia * sin(d.psi) ** 2 * d.phi_d1


def _get_dumbbell_observables(system: System) -> list[ObservableSpec]:
    """Return the energy and the angular momentum of the dumbbell."""
    del system
    return [
        ObservableSpec('energy', _get_dumbbell_energy, element='dumbbell'),
        ObservableSpec('angular_momentum', _get_dumbbell_momentum, element='dumbbell'),
    ]


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'gravity', 'spin-orbit coupling', 'lagrangian'),
    references=('Dilão and Murteira 2020', 'Dilão 2023'),
    defaults={
        'until': 60.0,
        'dt': 0.05,
        'solver': ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12),
        'invariants': {'dumbbell.energy': 1e-6, 'dumbbell.angular_momentum': 1e-6},
    },
    observables=_get_dumbbell_observables,
)
def keplerian_dumbbell(
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
) -> System:
    """Return the Keplerian dumbbell: two masses on a rigid massless arm that orbit a primary under gravity.

    The centre of mass of the dumbbell at the cylindrical coordinates `(u, theta, v)` about the primary and the arm at
    the spherical angles `(phi, psi)` move under the dimensionless Lagrangian
    `L = (u'² + u² theta'² + v'²) / 2 + I (psi'² + sin² psi phi'²) / 2 + (1 - delta) / rho12 + delta / rho13`, with the
    moment of inertia `I = delta (1 - delta) epsilon² / mu` of the arm. The distances of the two masses from the primary
    are `rho12² = u² + v² + delta² epsilon² + 2 delta epsilon P` and
    `rho13² = u² + v² + (1 - delta)² epsilon² - 2 (1 - delta) epsilon P`, with
    `P = u sin psi cos(theta - phi) + v cos psi`. Its Euler-Lagrange equations are the equations of motion of Dilão
    and Murteira. The gravity gradient across the arm couples the spin to the orbit, the simplest spin-orbit coupling
    of a rigid body. For `epsilon = 0` the orbit is a Kepler orbit, of period `2π` at `u = 1`. The energy and the
    angular momentum about the normal to the orbital plane are conserved. By default the arm spins three times faster
    than it orbits, in the orbital plane, and its spin and orbit exchange angular momentum.

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

    Returns:
        system:
            The system, with the coordinates `dumbbell.u`, `dumbbell.theta`, `dumbbell.v`, `dumbbell.phi`, and
            `dumbbell.psi` of order 2, and the observables `dumbbell.energy` and `dumbbell.angular_momentum`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import keplerian_dumbbell
        >>> keplerian_dumbbell(epsilon=0.2).elements[0].epsilon
        0.2
    """
    dumbbell = DumbbellElement(
        'dumbbell',
        u=u,
        u_d1=vu,
        theta=theta,
        theta_d1=omega,
        v=v,
        v_d1=vv,
        phi=phi,
        phi_d1=spin,
        psi=psi,
        psi_d1=vpsi,
        delta=delta,
        epsilon=epsilon,
        mu=mu,
    )
    motion = EulerLagrangeInteraction(
        'motion',
        target='dumbbell',
        states=COORDINATES,
        lagrangian=_get_dumbbell_lagrangian,
    )
    return System(elements=[dumbbell], interactions=[motion])


@register_system(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'gravity', 'general relativity', 'precession'),
    references=('Einstein 1915', 'Dilão 2023'),
    defaults={'until': 60.0, 'dt': 0.02},
)
def relativistic_orbit(u: float = 1.5, du: float = 0.0, *, alpha: float = 1.0, epsilon: float = 0.05) -> System:
    """Return Einstein's orbit equation `u'' + u = alpha + epsilon u²` of a planet, whose perihelion precesses.

    The inverse radius `u = 1 / r` of the planet is a function of its polar angle, which the simulation time stands
    for, so the orbit in the plane is `(cos t / u, sin t / u)`. For `epsilon = 0` the orbit is a closed Newtonian
    ellipse of eccentricity `|u(0) / alpha - 1|` for `u'(0) = 0`. The term `epsilon u²` is the first relativistic
    correction, and with it the orbit is a rosette whose perihelion advances by about `2π epsilon alpha` every
    revolution. The equation has two fixed points, the roots `(1 ∓ sqrt(1 - 4 alpha epsilon)) / (2 epsilon)` of
    `epsilon u² - u + alpha`. The smaller one, close to `alpha`, is a centre, the stable circular orbit, and the
    larger one is a saddle, the unstable circular orbit close to the black hole, beyond which a planet falls in.

    Args:
        u:
            The initial inverse radius.

        du:
            The initial derivative of the inverse radius along the angle.

        alpha:
            The Newtonian inverse semi-latus rectum.

        epsilon:
            The strength of the relativistic correction.

    Returns:
        system:
            The system, with the inverse radius `planet.u` of order 2 along the angle and its derivative
            `planet.u_d1`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import relativistic_orbit
        >>> relativistic_orbit(epsilon=0.0).elements[0].epsilon
        0.0
    """
    return System(elements=[RelativisticOrbitElement('planet', u=u, u_d1=du, alpha=alpha, epsilon=epsilon)])
