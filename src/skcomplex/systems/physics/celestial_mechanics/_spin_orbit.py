"""Celestial mechanics systems of spin-orbit coupling and relativistic orbits.

The module holds the spin-orbit coupling of a dumbbell and relativistic orbits.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ScipyODESolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.mechanics import LagrangianInteraction

COORDINATES = ('u', 'theta', 'v', 'phi', 'psi')


class DumbbellElement(Element):
    """Rigid dumbbell orbiting a primary.

    The element holds the position of its centre of mass and the orientation of its arm.

    The cylindrical coordinates `(u, theta, v)` place the centre of mass of the dumbbell relative to the primary, and
    the spherical angles `(phi, psi)` give the direction of its arm, of length `epsilon` in units of the mean orbit
    radius. The mass `1 - delta` of the dumbbell sits at `delta epsilon` along the arm and the mass `delta` at
    `1 - delta` against it. The five coordinates are those `DumbbellMotionInteraction` moves, and their velocities are
    the derivatives `u'`, `theta'`, `v'`, `phi'`, and `psi'`.

    Args:
        label:
            The label of the dumbbell.

        u:
            The initial distance from the axis through the primary.

        theta:
            The initial orbital angle.

        v:
            The initial height above the orbital plane.

        phi:
            The initial azimuth of the arm.

        psi:
            The initial polar angle of the arm, `pi / 2` in the orbital plane.

        delta:
            The share of the second mass in the mass of the dumbbell, `1 / 2` for equal masses.

        epsilon:
            The length of the arm in units of the mean orbit radius.

        mu:
            The share of the primary in the mass of the three bodies.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DumbbellElement
        >>> DumbbellElement('dumbbell').delta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        u: float = 1.0,
        theta: float = 0.0,
        v: float = 0.0,
        phi: float = 0.0,
        psi: float = math.pi / 2,
        delta: Annotated[float, RealSet.open(0, 1)] = 0.5,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        mu: Annotated[float, RealSet.open(0, 1)] = 0.9,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.u = u
        self.theta = theta
        self.v = v
        self.phi = phi
        self.psi = psi
        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu


def _get_dumbbell_rates(d: SimpleNamespace) -> SimpleNamespace:
    """Return the velocities of the coordinates of a dumbbell, the values named with a prime."""
    return SimpleNamespace(**{name: getattr(d, name).diff(d.t) for name in COORDINATES})


def _get_dumbbell_energy(d: SimpleNamespace) -> Expression:
    """Return the energy of a dumbbell, the sum of its kinetic and potential energies."""
    kinetic, potential = _get_dumbbell_parts(d, _get_dumbbell_rates(d))
    return kinetic + potential


def _get_dumbbell_momentum(d: SimpleNamespace) -> Expression:
    """Return the angular momentum of a dumbbell about the normal of the orbital plane."""
    rates = _get_dumbbell_rates(d)
    inertia = d.delta * (1 - d.delta) / d.mu * d.epsilon**2
    return d.u**2 * rates.theta + inertia * sin(d.psi) ** 2 * rates.phi


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


class DumbbellMotionInteraction(LagrangianInteraction):
    """Motion of a rigid dumbbell around a primary.

    The motion is the Euler-Lagrange equations of the Lagrangian of its orbit and its spin.

    The Lagrangian is `L = (u'² + u² theta'² + v'²) / 2 + I (psi'² + sin² psi phi'²) / 2 + (1 - delta) / rho12 + delta /
    rho13`, with the moment of inertia `I = delta (1 - delta) epsilon² / mu` of the arm and the distances `rho12` and
    `rho13` of the two masses from the primary, all read from the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the dumbbell.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DumbbellMotionInteraction
        >>> DumbbellMotionInteraction('motion', target='dumbbell').target
        'dumbbell'
    """

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the coordinates `(u, theta, v, phi, psi)` of the dumbbell."""
        return [getattr(context.target, name) for name in COORDINATES]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian of the orbit and the spin of the dumbbell.

        Args:
            context:
                The namespace of the dumbbell, the time `t`, and the parameters.

            positions:
                The symbols of the coordinates.

            velocities:
                The symbols of the velocities.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        target = context.target
        d = SimpleNamespace(
            **dict(zip(COORDINATES, positions, strict=True)),
            delta=target.delta,
            epsilon=target.epsilon,
            mu=target.mu,
        )
        rates = SimpleNamespace(**dict(zip(COORDINATES, velocities, strict=True)))
        kinetic, potential = _get_dumbbell_parts(d, rates)
        return kinetic - potential


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
    `dumbbell.u`, `dumbbell.theta`, `dumbbell.v`, `dumbbell.phi`, and `dumbbell.psi`, their velocities named with a
    prime, and the observables `dumbbell.energy` and `dumbbell.angular_momentum`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    invariants: ClassVar[Mapping[str, float]] = {'dumbbell.energy': 1e-6, 'dumbbell.angular_momentum': 1e-6}
    observables = (
        Observable('energy', _get_dumbbell_energy, element='dumbbell'),
        Observable('angular_momentum', _get_dumbbell_momentum, element='dumbbell'),
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two masses on a rigid arm that orbit a primary.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        dumbbell = DumbbellElement(
            'dumbbell',
            u=self.u,
            theta=self.theta,
            v=self.v,
            phi=self.phi,
            psi=self.psi,
            delta=self.delta,
            epsilon=self.epsilon,
            mu=self.mu,
            values={"u'": self.vu, "theta'": self.omega, "v'": self.vv, "phi'": self.spin, "psi'": self.vpsi},
        )
        motion = DumbbellMotionInteraction('motion', target='dumbbell')
        solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
        self.system_ = System(self.label, elements=[dumbbell], interactions=[motion], solver=solver)


class EinsteinOrbitInteraction(Interaction):
    """Einstein's orbit equation `u'' = alpha + epsilon u² - u` of the inverse radius of a planet.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the planet, whose value `u` is the inverse radius.

        alpha:
            The Newtonian inverse semi-latus rectum.

        epsilon:
            The strength of the relativistic correction.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import EinsteinOrbitInteraction
        >>> EinsteinOrbitInteraction('orbit', target='planet', epsilon=0.1).epsilon
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: Annotated[float, RealSet.open(0, oo)] = 1.0,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.epsilon = epsilon

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the second derivative of the inverse radius along the angle.

        Args:
            context:
                The namespace of the planet, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = context.target.u
            self.laws_ = [RateLaw(u.diff(context.t, 2), context.alpha + context.epsilon * u**2 - u)]


class RelativisticOrbitSystem(BaseSystem):
    """Einstein's orbit equation `u'' + u = alpha + epsilon u²` of a planet.

    The perihelion of the planet precesses.

    The inverse radius `u = 1 / r` of the planet is a function of its polar angle, which the simulation time stands for,
    so the orbit in the plane is `(cos t / u, sin t / u)`. For `epsilon = 0` the orbit is a closed Newtonian ellipse of
    eccentricity `|u(0) / alpha - 1|` for `u'(0) = 0`. The term `epsilon u²` is the first relativistic correction, and
    with it the orbit is a rosette whose perihelion advances by about `2π epsilon alpha` every revolution. The equation
    has two fixed points, the roots `(1 ∓ sqrt(1 - 4 alpha epsilon)) / (2 epsilon)` of `epsilon u² - u + alpha`. The
    smaller one, close to `alpha`, is a centre, the stable circular orbit, and the larger one is a saddle, the unstable
    circular orbit close to the black hole, beyond which a planet falls in. The system has the inverse radius `planet.u`
    along the angle and its derivative `planet.u'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RelativisticOrbitSystem
        >>> trajectory = RelativisticOrbitSystem(u=1.5).prepare().simulate(until=1.0)
        >>> float(trajectory.get_values('planet.u')[0])
        1.5
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.u = u
        self.du = du
        self.alpha = alpha
        self.epsilon = epsilon

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the inverse radius of a planet along its polar angle.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        planet = Element('planet', values={'u': self.u, "u'": self.du})
        orbit = EinsteinOrbitInteraction('orbit', target='planet', alpha=self.alpha, epsilon=self.epsilon)
        self.system_ = System(self.label, elements=[planet], interactions=[orbit])
