"""Cosmologies from the Einstein equations of a metric ansatz.

The module covers a homogeneous universe and a collapsing ball of dust.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import QQ, SR, Expression, diagonal_matrix, function, oo, sin
from sage.sets.real_set import RealSet
from sage.structure.element import Matrix

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import EinsteinInteraction
from ....core.components.spaces import LatticeSpace, PseudoRiemannianSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

COUPLING = 8 * math.pi
"""The coupling `κ = 8 π G / c⁴` of the Einstein equations in units with `G = c = 1`."""

COORDINATES = ('t', 'r', 'theta', 'phi')
"""The comoving coordinates of the spherical ansatzes."""


class RobertsonWalkerSpace(PseudoRiemannianSpace):
    """Homogeneous and isotropic spacetime of the Robertson-Walker ansatz.

    The scale factor `a(t)` is unknown.

    The line element in the comoving coordinates `(t, r, theta, phi)` is
    `ds² = -dt² + a(t)² (dr² / (1 - k r²) + r² (dθ² + sin²θ dφ²))`, with the spatial curvature `k`. The scale factor
    is the function `a` that `EinsteinInteraction` solves for, the state `a` of its target.

    Args:
        k:
            The spatial curvature, negative for an open, zero for a flat, and positive for a closed universe.

        label:
            The label of the space in its system.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import RobertsonWalkerSpace
        >>> space = RobertsonWalkerSpace(k=0.0).prepare()
        >>> (space.inverse_metric_ * space.einstein_)[0, 0]
        -3*diff(a(t), t)^2/a(t)^2
    """

    def __init__(self, k: float = 0.0, *, label: str = 'space') -> None:
        super().__init__(curvature=True, label=label)
        self.k = k

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in COORDINATES)

    def _get_metric(self) -> Matrix:
        """Return the Robertson-Walker metric of the scale factor `a(t)`."""
        t, r, theta, _ = self._get_coordinates()
        a = function('a')(t)
        return diagonal_matrix(SR, [-1, a**2 / (1 - QQ(self.k) * r**2), a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])


class LemaitreTolmanBondiSpace(PseudoRiemannianSpace):
    """Spherical spacetime of dust in comoving coordinates.

    This is the Lemaître-Tolman-Bondi ansatz.

    The line element in `(t, r, theta, phi)` is
    `ds² = -dt² + (∂R/∂r)² dr² / (1 + 2 E(r)) + R(t, r)² (dθ² + sin²θ dφ²)`, where the areal radius `R(t, r)` of the
    shell of comoving radius `r` is the unknown and `E(r)` is the energy of the shell per unit mass. They are the
    functions `R` and `energy`, a state and a parameter of the target of `EinsteinInteraction`.

    Args:
        label:
            The label of the space in its system.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import LemaitreTolmanBondiSpace
        >>> space = LemaitreTolmanBondiSpace().prepare()
        >>> space.metric_[2, 2]
        R(t, r)^2
    """

    def __init__(self, *, label: str = 'space') -> None:
        super().__init__(curvature=True, label=label)

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in COORDINATES)

    def _get_metric(self) -> Matrix:
        """Return the Lemaître-Tolman-Bondi metric.

        The metric is that of the areal radius `R(t, r)` and the energy `E(r)`.
        """
        t, r, theta, _ = self._get_coordinates()
        radius = function('R')(t, r)
        energy = function('energy')(r)
        return diagonal_matrix(
            SR,
            [-1, radius.diff(r) ** 2 / (1 + 2 * energy), radius**2, radius**2 * sin(theta) ** 2],
        )


class HomogeneousUniverseElement(Element):
    """Homogeneous universe of matter, radiation, and vacuum energy.

    The universe has the scale factor `a` of order 2.

    The densities are measured by their parameters `Ω` today, at `a = 1`, relative to the critical density
    `3 H0² / κ`: matter dilutes as `a⁻³`, radiation as `a⁻⁴`, and the vacuum energy of a cosmological constant stays.

    Args:
        label:
            The label of the universe.

        visual:
            How the universe is drawn, or `None` for none.

        a:
            The initial scale factor.

        a_d1:
            The initial rate of change of the scale factor.

        H0:
            The Hubble constant today.

        omega_matter:
            The density parameter of matter.

        omega_radiation:
            The density parameter of radiation.

        omega_lambda:
            The density parameter of the cosmological constant.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import HomogeneousUniverseElement
        >>> HomogeneousUniverseElement('universe', omega_radiation=1e-4).omega_radiation
        0.0001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.open(0, oo), order=2)] = 1.0,
        a_d1: float = 1.0,
        H0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        omega_matter: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        omega_radiation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        omega_lambda: Annotated[float, ParameterSpec()] = 0.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.a_d1 = a_d1
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_radiation = omega_radiation
        self.omega_lambda = omega_lambda


class DustShellElement(Element):
    """Shell of a spherical ball of dust.

    The shell has the areal radius `R` of order 2 and the energy per unit mass `energy`.

    Args:
        label:
            The label of the shell.

        visual:
            How the shell is drawn, or `None` for none.

        R:
            The initial areal radius.

        R_d1:
            The initial rate of change of the areal radius.

        energy:
            The energy per unit mass, negative for a bound shell.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import DustShellElement
        >>> DustShellElement('shell', R=2.0).R
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        R: Annotated[float, StateSpec(domain=RealSet.open(0, oo), order=2)] = 1.0,
        R_d1: float = 0.0,
        energy: Annotated[float, ParameterSpec(domain=RealSet.open(-0.5, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.R = R
        self.R_d1 = R_d1
        self.energy = energy


class FLRWCosmologySystem(BaseSystem):
    """A homogeneous universe whose expansion follows from the Einstein equations.

    The equations are those of the Robertson-Walker metric.

    The metric of `RobertsonWalkerSpace` gives, through its Einstein tensor, the two Friedmann equations of a perfect
    fluid of density `rho` and pressure `p`. The space-space component `G^r_r = -(2 a a'' + a'² + k) / a² = κ p` is the
    law of the scale factor, and the time-time component `G^t_t = -3 (a'² + k) / a² = -κ rho` is the constraint `(a'/a)²
    = κ rho / 3 - k / a²`, which sets the initial rate and which the evolution keeps. The fluid holds matter of pressure
    0, radiation of pressure `rho / 3`, and the vacuum energy of a cosmological constant of pressure `-rho`, with the
    density parameters today, at `a = 1`, and the curvature `k = -Ω_k H0²` of the remainder `Ω_k = 1 - Ω_m - Ω_r - Ω_Λ`.
    Units have `G = c = 1`. The system has the scale factor `universe.a` of order 2 and its rate `universe.a_d1`.

    Args:
        a:
            The initial scale factor.

        H0:
            The Hubble constant, the expansion rate at `a = 1`.

        omega_matter:
            The density parameter of matter.

        omega_radiation:
            The density parameter of radiation.

        omega_lambda:
            The density parameter of the cosmological constant.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The universe does not expand at the initial scale factor.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import FLRWCosmologySystem
        >>> FLRWCosmologySystem(1.0, omega_matter=1.0, omega_lambda=0.0).prepare().system_.elements[0].a_d1
        1.0
    """

    field = 'physics/cosmology'
    topics = ('cosmology', 'general relativity', 'ode', 'second order', 'gravity', 'dark energy')
    references = ('Schwichtenberg 2020', 'Satz 2022', 'Friedmann 1922', 'Lemaître 1927', 'Carroll 2004')
    until = 2.0
    dt = 0.02
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)

    def __init__(
        self,
        a: float = 0.3,
        *,
        H0: float = 1.0,
        omega_matter: float = 0.3,
        omega_radiation: float = 0.0,
        omega_lambda: float = 0.7,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_radiation = omega_radiation
        self.omega_lambda = omega_lambda

    @staticmethod
    def _get_perfect_fluid(s: SimpleNamespace) -> Matrix:
        """Return the mixed energy-momentum tensor `diag(-rho, p, p, p)`.

        The tensor is that of matter, radiation, and vacuum at rest.
        """
        universe = s.target
        critical = 3 * universe.H0**2 / COUPLING
        matter = universe.omega_matter / universe.a**3
        radiation = universe.omega_radiation / universe.a**4
        density = critical * (matter + radiation + universe.omega_lambda)
        pressure = critical * (radiation / 3 - universe.omega_lambda)
        return diagonal_matrix(SR, [-density, pressure, pressure, pressure])

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        curvature = 1 - self.omega_matter - self.omega_radiation - self.omega_lambda
        square = self.H0**2 * (
            self.omega_matter / self.a + self.omega_radiation / self.a**2 + self.omega_lambda * self.a**2 + curvature
        )
        if square < 0:
            error_msg = (
                f'The Friedmann constraint gives a negative squared expansion rate {square} at a = {self.a}. Choose a '
                'and the density parameters for which it is positive.'
            )
            raise ValidationError(error_msg)
        universe = HomogeneousUniverseElement(
            'universe',
            a=self.a,
            a_d1=math.sqrt(square),
            H0=self.H0,
            omega_matter=self.omega_matter,
            omega_radiation=self.omega_radiation,
            omega_lambda=self.omega_lambda,
        )
        einstein = EinsteinInteraction(
            'einstein',
            target='universe',
            states='a',
            space=RobertsonWalkerSpace(k=-curvature * self.H0**2),
            components=[(1, 1)],
            source=self._get_perfect_fluid,
            coupling=COUPLING,
        )
        return System(self.label, elements=[universe], interactions=[einstein])


class OppenheimerSnyderCollapseSystem(BaseSystem):
    """A ball of dust at rest that collapses under its own gravity into a black hole.

    The collapse follows Oppenheimer and Snyder.

    The ball is a field of comoving shells on a radial lattice, the Lemaître-Tolman-Bondi ansatz of
    `LemaitreTolmanBondiSpace` in units with `G = c = 1`. Dust has no pressure, so the radial component of the Einstein
    equations, `G^r_r = -(R'² + 2 R R'' - 2 E) / R² = 0` with `'` the derivative by proper time, is the law of the areal
    radius `R` of every shell, and its first integral `R'² = 2 M(r) / R + 2 E(r)`, from `G^t_t`, holds the mass `M(r)`
    inside the shell. A uniform ball of mass `M` and radius `R₀` at rest has `M(r) = M (r / R₀)³` and `E(r) = -M(r) /
    r`, so every shell follows the same cycloid `R = r (1 + cos η) / 2` with `t = (η + sin η) / (2 √k)` and `k = 2 M /
    R₀³`, as the interior of Oppenheimer and Snyder. The surface crosses its Schwarzschild radius `2 M` and the ball
    reaches zero size at the proper time `π / (2 √k)`. The system has the areal radius `ball.R` of every shell and its
    rate `ball.R_d1`.

    Args:
        n_shells:
            The number of shells, which split the comoving radius `R₀` evenly, the outermost at the surface.

        radius:
            The initial radius `R₀` of the ball.

        mass:
            The mass `M` of the ball.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The ball starts inside its Schwarzschild radius.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import OppenheimerSnyderCollapseSystem
        >>> round(float(OppenheimerSnyderCollapseSystem(4).prepare().system_.elements[0].values['energy'][-1]), 3)
        -0.1
    """

    field = 'physics/cosmology'
    topics = ('cosmology', 'general relativity', 'pde', 'second order', 'gravity', 'black holes')
    references = ('Schwichtenberg 2020', 'Satz 2022', 'Oppenheimer and Snyder 1939', 'Tolman 1934', 'Bondi 1947')
    until = 34.5
    dt = 0.1
    solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)

    def __init__(
        self,
        n_shells: int = 32,
        *,
        radius: float = 10.0,
        mass: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_shells = n_shells
        self.radius = radius
        self.mass = mass

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.radius <= 2 * self.mass:
            error_msg = (
                f'The radius {self.radius} is inside the Schwarzschild radius 2 M = {2 * self.mass}. Give a larger '
                'radius.'
            )
            raise ValidationError(error_msg)
        radii = self.radius * np.arange(1, self.n_shells + 1) / self.n_shells
        energies = -self.mass * radii**2 / self.radius**3
        ball = FieldElement(
            'ball',
            element=DustShellElement,
            space=LatticeSpace((self.n_shells,), spacing=self.radius / self.n_shells),
            values={'R': radii, 'energy': energies},
        )
        einstein = EinsteinInteraction(
            'einstein',
            target='ball',
            states='R',
            space=LemaitreTolmanBondiSpace(),
            components=[(1, 1)],
            coupling=COUPLING,
            axes={1: 0},
        )
        return System(self.label, elements=[ball], interactions=[einstein])
