"""Cosmologies from the Einstein equations of a metric ansatz.

The module covers a homogeneous universe and a collapsing ball of dust.

`COUPLING` holds the coupling `κ = 8 π G / c⁴` of the Einstein equations in units with `G = c = 1`. `COORDINATES` holds
the comoving coordinates of the spherical ansatzes. `ACCURATE` holds the solver of tight tolerances of the systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import QQ, SR, Expression, diagonal_matrix, function, matrix, oo, sin
from sage.sets.real_set import RealSet
from sage.structure.element import Matrix
from sage.symbolic.function import BuiltinFunction, SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import partial
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ScipyODESolver
from ....core.errors import SpaceError, ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.relativity import SpacetimeSpace

COUPLING = 8 * math.pi
ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
COORDINATES = ('t', 'r', 'theta', 'phi')
MAX_ORDER = 2


class EinsteinFieldEquationsInteraction(Interaction):
    """Einstein equations `G^mu_nu + Λ δ^mu_nu = κ T^mu_nu` of a metric ansatz.

    The equations are reduced to rate laws of the unknowns of the ansatz. The metric of the space is an ansatz: a
    symmetric matrix whose unknowns are symbolic functions of the coordinates, each named after a state or a parameter
    of the target, such as the scale factor `a(t)` of a homogeneous universe or the areal radius `R(t, r)` of the shells
    of a spherical dust ball. The interaction takes the mixed Einstein tensor `G^mu_nu = g^(mu lambda) G_(lambda nu)`
    the space derives, keeps one component `(mu, nu)` per state, and solves these equations, linear in the second time
    derivatives, for them, which gives one rate law per state of order 2.

    The time coordinate becomes the time of the system, a derivative of an unknown along it the time derivative of its
    state, and a derivative along a coordinate that `axes` maps to an axis of the lattice of a field the derivative
    `partial` of the field along that axis, so a spherical ansatz reduces to a field in `1 + 1` dimensions. An equation
    that keeps another coordinate, a mixed derivative in time and space, or a derivative above the second does not
    reduce to such laws and raises an error, as does a set of components that does not fix the second derivatives: the
    remaining components are constraints, such as the Friedmann equation, which the initial values satisfy and the laws
    keep.

    The source function takes the namespace of the interaction, with the namespace `target`, and returns the mixed
    energy-momentum tensor `T^mu_nu`, such as `diag(-rho, p, p, p)` of a perfect fluid at rest, in terms of the states
    and parameters of the target.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element or field whose states and parameters are the unknowns of the ansatz.

        states:
            The name of the state of order 2 that an equation determines, or the names of several states.

        space:
            The pseudo-Riemannian space of the ansatz, with the curvature.

        components:
            The index pair `(mu, nu)` of the mixed equation of every state, in the order of the states.

        source:
            A function of the namespace that returns the mixed energy-momentum tensor, or `None` for vacuum.

        coupling:
            The coupling `κ = 8 π G / c⁴` of the source.

        cosmological_constant:
            The cosmological constant `Λ`.

        time:
            The index of the time coordinate.

        axes:
            The axis of the lattice of the target along every spatial coordinate the unknowns depend on, by the index
            of the coordinate.

    Attributes:
        space_:
            The prepared space.

        equations_:
            The mixed Einstein tensor `G^mu_nu` of the space, whose kept components the laws simplify.

    Raises:
        ValidationError:
            A state is not a name, the components do not give one index pair per state, the source is not callable, or
            an equation does not reduce to rate laws of the states.

        SpaceError:
            The space is not a pseudo-Riemannian space with curvature, raised by `prepare`.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, sin
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.cosmology import EinsteinFieldEquationsInteraction
        >>> from skcomplex.systems.physics.common.relativity import SpacetimeSpace
        >>> t, r, theta, phi = SR.var('t r theta phi')
        >>> a = function('a')(t)
        >>> metric = diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])
        >>> flat = SpacetimeSpace(metric, (t, r, theta, phi))
        >>> vacuum = EinsteinFieldEquationsInteraction(
        ...     'vacuum', target='universe', states='a', space=flat, components=[(1, 1)], cosmological_constant=3.0
        ... )
        >>> universe = Element('universe', values={'a': 1.0, "a'": 1.0})
        >>> system = System(elements=[universe], interactions=[vacuum]).prepare()
        >>> trajectory = system.simulate(until=1.0, dt=0.001)
        >>> round(float(trajectory.get_values('universe.a')[-1]), 4)
        2.7183
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        states: str | Sequence[str] = 'x',
        space: SpacetimeSpace | None = None,
        components: Sequence[tuple[int, int]] = (),
        source: Callable[[SimpleNamespace], object] | None = None,
        coupling: float = 8 * math.pi,
        cosmological_constant: float = 0.0,
        time: int = 0,
        axes: Mapping[int, int] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.states = states
        self.space = space
        self.components = components
        self.source = source
        self.coupling = coupling
        self.cosmological_constant = cosmological_constant
        self.time = time
        self.axes = axes

    def _get_names(self) -> tuple[str, ...]:
        """Return the names of the states the equations determine."""
        return (self.states,) if isinstance(self.states, str) else tuple(self.states)

    def _check(self) -> None:
        """Raise an error for states not names, mismatched components, a source, or a space.

        Raises:
            SpaceError:
                The space is not a pseudo-Riemannian space with curvature.
            ValidationError:
                A state is not a name, the components do not give one index pair per state, or the source is not
                callable.
        """
        names = self._get_names()
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'{type(self).__name__} {self.label!r} solves for the states {self.states!r}. Give names.'
            raise ValidationError(error_msg)
        pairs = [tuple(pair) for pair in self.components]
        if len(pairs) != len(names) or not all(len(pair) == MAX_ORDER for pair in pairs):
            error_msg = (
                f'{type(self).__name__} {self.label!r} has the components {self.components!r} for the states '
                f'{names!r}. Give one index pair (mu, nu) per state.'
            )
            raise ValidationError(error_msg)
        if self.source is not None and not callable(self.source):
            error_msg = f'The source of {self.label!r} is {self.source!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)
        if not isinstance(self.space, SpacetimeSpace) or not self.space.curvature:
            error_msg = (
                f'{type(self).__name__} {self.label!r} reads the curvature of {self.space!r}. Give a '
                '`SpacetimeSpace` with `curvature=True`.'
            )
            raise SpaceError(error_msg)

    def _check_arguments(self) -> None:
        """Raise an error for states not names, mismatched components, a source, or a space.

        Raises:
            SpaceError:
                The space is not a pseudo-Riemannian space with curvature.
            ValidationError:
                A state is not a name, the components do not give one index pair per state, or the source is not
                callable.
        """
        super()._check_arguments()
        self._check()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the prepared space, the mixed Einstein tensor of its metric, and the laws of a binding.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                An equation does not reduce to rate laws of the states.
        """
        super()._prepare(context)
        if not isinstance(self.space, SpacetimeSpace):
            return
        self.space_ = self.space.prepare()
        self.equations_: Matrix = self.space_.inverse_metric_ * self.space_.einstein_
        self.laws_ = self._get_rate_laws(context)

    def _walk(self, expression: Expression) -> Iterator[Expression]:
        """Yield an expression and every subexpression of it."""
        yield expression
        for operand in expression.operands():
            yield from self._walk(operand)

    def _get_unknown(self, member: SimpleNamespace, name: str) -> Expression:
        """Return the state or parameter of the target that a function of the ansatz names."""
        if not hasattr(member, name):
            error_msg = (
                f'The metric of {self.label!r} has the function {name!r}, which is no state or parameter of the '
                'target. Name every function of the ansatz after a state or a parameter.'
            )
            raise ValidationError(error_msg)
        return SR(getattr(member, name))

    def _get_functions(self) -> set[str]:
        """Return the names of the unknown functions of the ansatz."""
        return {
            item.operator().name()
            for entry in self.space_.metric_.list()
            for item in self._walk(SR(entry))
            if isinstance(item.operator(), SymbolicFunction) and not isinstance(item.operator(), BuiltinFunction)
        }

    def _get_derivative(
        self,
        item: Expression,
        member: SimpleNamespace,
        accelerations: Mapping[int, Mapping[str, Expression]],
    ) -> Expression:
        """Return the state derivative a derivative of a function of the ansatz stands for.

        Args:
            item:
                The derivative of a function of the coordinates.

            member:
                The namespace of the target.

            accelerations:
                The symbols of the first and the second time derivatives of every state, by the order and the name.

        Returns:
            value:
                A symbol of a time derivative of a state, or derivatives of the state along lattice axes.

        Raises:
            ValidationError:
                The derivative mixes coordinates, has an order above the second, or is along a coordinate without an
                axis.
        """
        kind = item.operator()
        name = kind.function().name()
        coordinates = self.space_.coordinates_
        along = [coordinates.index(item.operands()[index]) for index in kind.parameter_set()]
        order = len(along)
        axes = dict(self.axes or {})
        if len(set(along)) > 1 or order > MAX_ORDER:
            error_msg = (
                f'The equations of {self.label!r} hold the derivative {item}, which mixes coordinates or has an order '
                'above 2, so they do not reduce to rate laws. Choose other components or another ansatz.'
            )
            raise ValidationError(error_msg)
        if along[0] == self.time:
            if name not in accelerations[order]:
                error_msg = (
                    f'The equations of {self.label!r} hold the time derivative of {name!r}, which is not one of the '
                    f'states {self._get_names()!r}. Add it to the states.'
                )
                raise ValidationError(error_msg)
            return accelerations[order][name]
        if along[0] not in axes:
            error_msg = (
                f'The equations of {self.label!r} hold the derivative {item} along the coordinate '
                f'{coordinates[along[0]]}, which has no lattice axis. Map it to an axis in `axes`.'
            )
            raise ValidationError(error_msg)
        value = self._get_unknown(member, name)
        for _ in range(order):
            value = partial(value, SR(axes[along[0]]))
        return value

    def _translate(
        self,
        expression: Expression,
        context: SimpleNamespace,
        accelerations: Mapping[int, Mapping[str, Expression]],
    ) -> Expression:
        """Return an entry of the Einstein tensor in the values of the target.

        It checks that the entry reduces.

        Args:
            expression:
                The entry of the mixed Einstein tensor.

            context:
                The namespace of the target, the time `t`, and the parameters.

            accelerations:
                The symbols of the first and the second time derivatives of every state, by the order and the name.

        Returns:
            entry:
                The entry in the values of the target.

        Raises:
            ValidationError:
                The entry keeps a coordinate other than time, or a derivative does not reduce.
        """
        member = context.target
        derivatives = {
            item: self._get_derivative(item, member, accelerations)
            for item in self._walk(expression)
            if isinstance(item.operator(), FDerivativeOperator)
        }
        names = self._get_functions()
        functions = {
            item: self._get_unknown(member, item.operator().name())
            for item in self._walk(expression)
            if isinstance(item.operator(), SymbolicFunction) and item.operator().name() in names
        }
        expression = expression.subs(derivatives).subs(functions)
        coordinates = self.space_.coordinates_
        expression = expression.subs({coordinates[self.time]: SR(context.t)})
        kept = [item for index, item in enumerate(coordinates) if index != self.time and expression.has(item)]
        if kept:
            error_msg = (
                f'The equations of {self.label!r} keep the coordinates {kept}, so the ansatz does not reduce them. '
                'Choose components that are free of them, such as mixed diagonal components of a symmetric ansatz.'
            )
            raise ValidationError(error_msg)
        return expression

    def _get_source(self, context: SimpleNamespace) -> Matrix:
        """Return the mixed energy-momentum tensor, zero without a source."""
        n_dim = len(self.space_.coordinates_)
        if self.source is None:
            return matrix(SR, n_dim, n_dim)
        return matrix(SR, self.source(context))

    def _get_rate_laws(self, context: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate law of the second derivative of every state.

        It is solved from the chosen equations.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Returns:
            laws:
                The rate laws of the states.

        Raises:
            ValidationError:
                An equation does not reduce, is not linear in the second derivatives, or the equations do not fix them.
        """
        names = self._get_names()
        member = context.target
        t = context.t
        accelerations = {
            order: {name: SR.var(f'_derivative_{order}_{index}') for index, name in enumerate(names)}
            for order in (1, MAX_ORDER)
        }
        unknowns = list(accelerations[MAX_ORDER].values())
        velocities = {accelerations[1][name]: SR(getattr(member, name)).diff(t) for name in names}
        source = self._get_source(context)
        rows, rest = [], []
        for mu, nu in (tuple(pair) for pair in self.components):
            curvature = self._translate(SR(self.equations_[mu, nu]).simplify_full(), context, accelerations)
            vacuum = self.cosmological_constant if mu == nu else 0
            equation = curvature + vacuum - self.coupling * SR(source[mu, nu])
            coefficients = [equation.diff(unknown) for unknown in unknowns]
            if any(coefficient.has(unknown) for coefficient in coefficients for unknown in unknowns):
                error_msg = f'The equation {(mu, nu)} of {self.label!r} is not linear in the second derivatives.'
                raise ValidationError(error_msg)
            rows.append(coefficients)
            rest.append(-equation.subs(dict.fromkeys(unknowns, 0)))
        system = matrix(SR, rows)
        if SR(system.det()).simplify_full().is_zero():
            error_msg = (
                f'The components {self.components!r} of {self.label!r} do not fix the second derivatives of '
                f'{names!r}. Choose components that hold them, such as spatial diagonal ones.'
            )
            raise ValidationError(error_msg)
        rates = system.solve_right(matrix(SR, len(rest), 1, rest), check=False).column(0)
        return [
            RateLaw(SR(getattr(member, name)).diff(t, 2), SR(rate).simplify_full().subs(velocities))
            for name, rate in zip(names, rates, strict=True)
        ]


class RobertsonWalkerSpace(SpacetimeSpace):
    """Homogeneous and isotropic spacetime of the Robertson-Walker ansatz.

    The scale factor `a(t)` is unknown.

    The line element in the comoving coordinates `(t, r, theta, phi)` is
    `ds² = -dt² + a(t)² (dr² / (1 - k r²) + r² (dθ² + sin²θ dφ²))`, with the spatial curvature `k`. The scale factor
    is the function `a` that `EinsteinFieldEquationsInteraction` solves for, the state `a` of its target.

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


class LemaitreTolmanBondiSpace(SpacetimeSpace):
    """Spherical spacetime of dust in comoving coordinates.

    This is the Lemaître-Tolman-Bondi ansatz.

    The line element in `(t, r, theta, phi)` is
    `ds² = -dt² + (∂R/∂r)² dr² / (1 + 2 E(r)) + R(t, r)² (dθ² + sin²θ dφ²)`, where the areal radius `R(t, r)` of the
    shell of comoving radius `r` is the unknown and `E(r)` is the energy of the shell per unit mass. They are the
    functions `R` and `energy`, a state and a parameter of the target of `EinsteinFieldEquationsInteraction`.

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

    The universe has the scale factor `a`, and its rate is the derivative `a'`.

    The densities are measured by their parameters `Ω` today, at `a = 1`, relative to the critical density
    `3 H0² / κ`: matter dilutes as `a⁻³`, radiation as `a⁻⁴`, and the vacuum energy of a cosmological constant stays.

    Args:
        label:
            The label of the universe.

        a:
            The initial scale factor.

        H0:
            The Hubble constant today.

        omega_matter:
            The density parameter of matter.

        omega_radiation:
            The density parameter of radiation.

        omega_lambda:
            The density parameter of the cosmological constant.

        values:
            Further values by name, such as the initial rate `a'` of the scale factor, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import HomogeneousUniverseElement
        >>> HomogeneousUniverseElement('universe', omega_radiation=1e-4).omega_radiation
        0.0001
    """

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        H0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        omega_matter: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        omega_radiation: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        omega_lambda: float = 0.7,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.a = a
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_radiation = omega_radiation
        self.omega_lambda = omega_lambda


class DustShellElement(Element):
    """Shell of a spherical ball of dust.

    The shell has the areal radius `R` and the energy per unit mass `energy`, and the rate of the radius is the
    derivative `R'`. On a space, the element holds one shell per site.

    Args:
        label:
            The label of the shell.

        R:
            The initial areal radius, or one radius per site.

        energy:
            The energy per unit mass, negative for a bound shell, or one energy per site.

        space:
            The space whose sites hold the shells, or `None` for a single shell.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import DustShellElement
        >>> DustShellElement('shell', R=2.0).R
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        R: Annotated[float | npt.ArrayLike, RealSet.open(0, oo)] = 1.0,
        energy: Annotated[float | npt.ArrayLike, RealSet.open(-0.5, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.R = R
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
    Units have `G = c = 1`. The system has the scale factor `universe.a` and its rate `universe.a'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The universe does not expand at the initial scale factor.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import FLRWCosmologySystem
        >>> system = FLRWCosmologySystem(1.0, omega_matter=1.0, omega_lambda=0.0).prepare()
        >>> float(system.simulate(until=0.1).get_values("universe.a'")[0])
        1.0
    """

    field = 'physics/cosmology'
    topics = ('cosmology', 'general relativity', 'ode', 'second order', 'gravity', 'dark energy')
    references = ('Schwichtenberg 2020', 'Satz 2022', 'Friedmann 1922', 'Lemaître 1927', 'Carroll 2004')
    until = 2.0
    dt = 0.02

    def __init__(
        self,
        a: float = 0.3,
        *,
        H0: float = 1.0,
        omega_matter: float = 0.3,
        omega_radiation: float = 0.0,
        omega_lambda: float = 0.7,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a homogeneous universe whose expansion follows from the Einstein equations.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
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
            H0=self.H0,
            omega_matter=self.omega_matter,
            omega_radiation=self.omega_radiation,
            omega_lambda=self.omega_lambda,
            values={"a'": math.sqrt(square)},
        )
        einstein = EinsteinFieldEquationsInteraction(
            'einstein',
            target='universe',
            states='a',
            space=RobertsonWalkerSpace(k=-curvature * self.H0**2),
            components=[(1, 1)],
            source=self._get_perfect_fluid,
            coupling=COUPLING,
        )
        self.system_ = System(self.label, elements=[universe], interactions=[einstein], solver=ACCURATE)


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
    rate `ball.R'`.

    Args:
        n_shells:
            The number of shells, which split the comoving radius `R₀` evenly, the outermost at the surface.

        radius:
            The initial radius `R₀` of the ball.

        mass:
            The mass `M` of the ball.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The ball starts inside its Schwarzschild radius.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import OppenheimerSnyderCollapseSystem
        >>> round(float(OppenheimerSnyderCollapseSystem(4).prepare().system_.elements[0].energy[-1]), 3)
        -0.1
    """

    field = 'physics/cosmology'
    topics = ('cosmology', 'general relativity', 'pde', 'second order', 'gravity', 'black holes')
    references = ('Schwichtenberg 2020', 'Satz 2022', 'Oppenheimer and Snyder 1939', 'Tolman 1934', 'Bondi 1947')
    until = 34.5
    dt = 0.1

    def __init__(
        self,
        n_shells: int = 32,
        *,
        radius: float = 10.0,
        mass: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_shells = n_shells
        self.radius = radius
        self.mass = mass

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ball of dust shells that collapses under its own gravity.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        if self.radius <= 2 * self.mass:
            error_msg = (
                f'The radius {self.radius} is inside the Schwarzschild radius 2 M = {2 * self.mass}. Give a larger '
                'radius.'
            )
            raise ValidationError(error_msg)
        radii = self.radius * np.arange(1, self.n_shells + 1) / self.n_shells
        energies = -self.mass * radii**2 / self.radius**3
        ball = DustShellElement(
            'ball',
            R=radii,
            energy=energies,
            space=LatticeSpace((self.n_shells,), spacing=self.radius / self.n_shells),
        )
        einstein = EinsteinFieldEquationsInteraction(
            'einstein',
            target='ball',
            states='R',
            space=LemaitreTolmanBondiSpace(),
            components=[(1, 1)],
            coupling=COUPLING,
            axes={1: 0},
        )
        self.system_ = System(self.label, elements=[ball], interactions=[einstein], solver=ACCURATE)
