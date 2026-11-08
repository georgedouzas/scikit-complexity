"""Define Einstein interactions.

They are the field equations of a metric reduced by an ansatz to equations of its unknowns.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, matrix
from sage.structure.element import Matrix
from sage.symbolic.function import BuiltinFunction, SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator

from ...errors import SpaceError, ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RateLaw
from ..laws.utils import derivative, partial
from ..spaces import BaseSpace, PseudoRiemannianSpace
from ._interaction import Interaction

MAX_ORDER = 2


class EinsteinInteraction(Interaction):
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
            The label of the interaction, unique among the parts of its system.

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
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, sin
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import EinsteinInteraction
        >>> from skcomplex.core.components.spaces import PseudoRiemannianSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class UniverseElement(Element):
        ...     def __init__(self, label, a: Annotated[float, StateSpec(order=2)] = 1.0, a_d1: float = 0.0):
        ...         super().__init__(label)
        ...         self.a = a
        ...         self.a_d1 = a_d1
        >>> t, r, theta, phi = SR.var('t r theta phi')
        >>> a = function('a')(t)
        >>> metric = diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])
        >>> flat = PseudoRiemannianSpace(metric, (t, r, theta, phi))
        >>> vacuum = EinsteinInteraction(
        ...     'vacuum', target='universe', states='a', space=flat, components=[(1, 1)], cosmological_constant=3.0
        ... )
        >>> system = System(elements=[UniverseElement('universe', a_d1=1.0)], interactions=[vacuum])
        >>> trajectory = simulate(system, until=1.0, dt=0.001)
        >>> round(float(trajectory.values('universe.a')[-1]), 4)
        2.7183
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('states',))],
        states: str | Sequence[str] = 'x',
        space: PseudoRiemannianSpace | None = None,
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

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set the prepared space and the mixed Einstein tensor of its metric.

        It first raises an error for states not names, mismatched components, or a source.

        Args:
            context:
                The space the interaction inherits from its system, which the equations do not read.

        Raises:
            SpaceError:
                The space is not a pseudo-Riemannian space with curvature.
            ValidationError:
                The type declares states, a state is not a name, the components do not give one index pair per state,
                or the source is not callable.
        """
        super()._prepare(context)
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
        if not isinstance(self.space, PseudoRiemannianSpace) or not self.space.curvature:
            error_msg = (
                f'{type(self).__name__} {self.label!r} reads the curvature of {self.space!r}. Give a '
                '`PseudoRiemannianSpace` with `curvature=True`.'
            )
            raise SpaceError(error_msg)
        self.space_ = self.space.prepare()
        self.equations_: Matrix = self.space_.inverse_metric_ * self.space_.einstein_

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
        namespace: SimpleNamespace,
        accelerations: Mapping[int, Mapping[str, Expression]],
    ) -> Expression:
        """Return an entry of the Einstein tensor in the symbols of the namespace.

        It checks that the entry reduces.

        Args:
            expression:
                The entry of the mixed Einstein tensor.

            namespace:
                The namespace of the symbols the laws can use, with the namespace of the target.

            accelerations:
                The symbols of the first and the second time derivatives of every state, by the order and the name.

        Returns:
            entry:
                The entry in the symbols of the namespace.

        Raises:
            ValidationError:
                The entry keeps a coordinate other than time, or a derivative does not reduce.
        """
        member = self._get_member(namespace, 'target')
        items = list(self._walk(expression))
        derivatives = {
            item: self._get_derivative(item, member, accelerations)
            for item in items
            if isinstance(item.operator(), FDerivativeOperator)
        }
        expression = expression.subs(derivatives)
        functions = {
            item: self._get_unknown(member, item.operator().name())
            for item in self._walk(expression)
            if isinstance(item.operator(), SymbolicFunction)
            and not isinstance(item.operator(), BuiltinFunction)
            and item.operator() not in (partial, derivative)
        }
        expression = expression.subs(functions).subs({self.space_.coordinates_[self.time]: SR(namespace.t)})
        kept = [item for item in self.space_.coordinates_ if expression.has(item)]
        if kept:
            error_msg = (
                f'The equations of {self.label!r} keep the coordinates {kept}, so the ansatz does not reduce them. '
                'Choose components that are free of them, such as mixed diagonal components of a symmetric ansatz.'
            )
            raise ValidationError(error_msg)
        return expression

    def _get_source(self, namespace: SimpleNamespace) -> Matrix:
        """Return the mixed energy-momentum tensor, zero without a source."""
        n_dim = len(self.space_.coordinates_)
        if self.source is None:
            return matrix(SR, n_dim, n_dim)
        return matrix(SR, self.source(namespace))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rate law of the second derivative of every state.

        It is solved from the chosen equations.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the target.

        Returns:
            laws:
                The rate laws of the states.

        Raises:
            ValidationError:
                An equation does not reduce, is not linear in the second derivatives, or the equations do not fix them.
        """
        names = self._get_names()
        member = self._get_member(namespace, 'target')
        accelerations = {
            order: {name: SR.var(f'_derivative_{order}_{index}') for index, name in enumerate(names)}
            for order in (1, MAX_ORDER)
        }
        unknowns = list(accelerations[MAX_ORDER].values())
        velocities = {accelerations[1][name]: derivative(getattr(member, name), 1) for name in names}
        source = self._get_source(namespace)
        rows, rest = [], []
        for mu, nu in (tuple(pair) for pair in self.components):
            curvature = self._translate(SR(self.equations_[mu, nu]).simplify_full(), namespace, accelerations)
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
            RateLaw(getattr(member, name), SR(rate).simplify_full().subs(velocities))
            for name, rate in zip(names, rates, strict=True)
        ]
