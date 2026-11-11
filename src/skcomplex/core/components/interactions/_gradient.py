"""Define the interaction that moves states down the gradient of a potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterator, Mapping, Sequence
from math import prod
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, matrix, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.symbolic.operators import add_vararg, mul_vararg

from ...enums import ElementKind
from ...errors import ValidationError
from ...specs import RoleSpec
from ..base import BaseComponent
from ..laws import BaseLaw, RateLaw
from ..laws.utils import laplacian, partial
from ..laws.utils._tensors import distribute_operators
from ..spaces import BaseSpace
from ..types import Target
from ._interaction import Interaction


class GradientInteraction(Interaction):
    """An interaction that moves states down the gradient of a potential.

    Every state `x` that the interaction moves gets the rate `-dU/dx / d` on its highest time derivative, where `U` is
    the potential and `d` the divisor of its member. On a state of order 1 this is a gradient flow `x' = -dU/dx / d`,
    with `d` a friction. On a state of order 2 it is Newton's law `x'' = -dU/dx / d`, with `d` a mass or an inertia. A
    vector state gets the gradient of every component.

    On a field, the potential is the density `f` of a functional `F`, the integral of `f` over space, and may read the
    field, its derivatives, and its Laplacian. With `variational` the field moves by the functional derivative of `F`,
    as in the Allen-Cahn equation. With `conserved` the rate is the Laplacian of that derivative, as in the
    Cahn-Hilliard equation, so the total of the field stays constant. The potential function takes a namespace with the
    roles, the `sources`, the time `t`, and the parameters a subclass declares.

    Unlike `RateInteraction`, which takes the rate itself, it takes a potential and derives the rates from it. Choose it
    when the dynamics descend an energy or a cost, so that one scalar function gives every rate.

    It appears in many fields. Particles move in a potential well, phases separate in an alloy, a neural network
    descends its loss, and cells sort by their adhesion energy.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'descent'`.

        target:
            The path of the element or group whose states move.

        states:
            The name of the state that moves, or the names of several states.

        potential:
            A function of the namespace that returns the potential, or the density of the functional of a field.

        sources:
            The paths of the elements or groups the potential reads but does not move.

        divisor:
            The name of the state or parameter of every moving member that divides its rate, a function of the
            namespace of the member that returns the divisor, a mapping from the name of every moving state to its
            own divisor, or `None`.

        variational:
            Whether the potential is the density of a functional of a field, moved by its functional derivative.

        conserved:
            Whether the rate is the Laplacian of the derivative of the potential, which conserves the total of a field.

    Raises:
        ValidationError:
            A state or the divisor is not a name, or the potential is not callable.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.specs import StateSpec
        >>> class AmountElement(Element):
        ...     def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.x = x
        >>> from skcomplex.core.components.interactions import GradientInteraction
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> well = GradientInteraction('well', target='walker', states='x', potential=lambda s: s.target.x**2 / 2)
        >>> system = CompositeSystem(elements=[AmountElement('walker', x=1.0)], interactions=[well])
        >>> trajectory = simulate(system, until=1.0, dt=0.01)
        >>> round(float(trajectory.get_values('walker.x')[-1]), 4)
        0.3679
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('states', 'divisor'))],
        states: str | Sequence[str] = 'x',
        potential: Callable[[SimpleNamespace], object] | None = None,
        sources: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = (),
        divisor: (
            str | Callable[[SimpleNamespace], object] | Mapping[str, str | Callable[[SimpleNamespace], object]] | None
        ) = None,
        variational: bool = False,
        conserved: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.states = states
        self.potential = potential
        self.sources = sources
        self.divisor = divisor
        self.variational = variational
        self.conserved = conserved

    def _get_names(self) -> tuple[str, ...]:
        """Return the names of the states the interaction moves."""
        return (self.states,) if isinstance(self.states, str) else tuple(self.states)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a state or divisor that is not a name, or a bad potential.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, a state is not a name, a divisor is neither a name nor a function, or the
                potential is not callable.
        """
        super()._prepare(context)
        names = self._get_names()
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'{type(self).__name__} {self.label!r} moves the states {self.states!r}. Give names of states.'
            raise ValidationError(error_msg)
        divisors = dict(self.divisor) if isinstance(self.divisor, Mapping) else dict.fromkeys(names, self.divisor)
        valid = all(
            item is None or callable(item) or (isinstance(item, str) and item.isidentifier())
            for item in divisors.values()
        )
        if not valid or not set(divisors) <= set(names):
            error_msg = (
                f'The divisor of {self.label!r} is {self.divisor!r}. Give the name of a state or parameter, a function '
                'of the namespace of the member, or a mapping from the moving states to them.'
            )
            raise ValidationError(error_msg)
        if self.potential is not None and not callable(self.potential):
            error_msg = f'The potential of {self.label!r} is {self.potential!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _get_potential(self, namespace: SimpleNamespace) -> object:
        """Return the potential.

        This is the hook a subclass overrides to derive it from its parameters.
        """
        if self.potential is None:
            error_msg = f'{type(self).__name__} {self.label!r} has no potential. Give a function of the namespace.'
            raise ValidationError(error_msg)
        return self.potential(namespace)

    def _get_divisor(self, member: SimpleNamespace, name: str) -> object:
        """Return the divisor of a moving state of a member, or `None` for none."""
        divisor = self.divisor.get(name) if isinstance(self.divisor, Mapping) else self.divisor
        if divisor is None:
            return None
        return getattr(member, divisor) if isinstance(divisor, str) else divisor(member)

    def _get_rate_state(self, member: SimpleNamespace, name: str) -> object:
        """Return the state whose highest derivative gets the rate of a moving state.

        It is the moving state itself.
        """
        return getattr(member, name)

    def _walk(self, expression: Expression) -> Iterator[Expression]:
        """Yield an expression and every subexpression of it."""
        yield expression
        for operand in expression.operands():
            yield from self._walk(operand)

    def _get_flux_divergence(
        self,
        fluxes: dict[int, Expression],
        derivatives: dict[int, Expression],
        field: Expression,
    ) -> Expression:
        """Return `sum_k partial(flux_k, k)`, or `c laplacian(x)` for fluxes `c x_k`.

        The Laplacian applies when every flux is `c x_k` along every axis read. The Laplacian keeps the compact stencil
        of the field, which nested first derivatives would widen.
        """
        coefficients = [SR(fluxes[axis] / derivatives[axis]) for axis in sorted(fluxes)]
        coefficient = coefficients[0]
        uniform = sorted(fluxes) == list(range(len(fluxes))) and all(
            item.is_trivially_equal(coefficient) for item in coefficients
        )
        constant = not coefficient.has(field) and not any(coefficient.has(item) for item in derivatives.values())
        if uniform and constant:
            return coefficient * laplacian(field)
        return sum((partial(flux, SR(axis)) for axis, flux in fluxes.items()), SR(0))

    def _get_variation(self, density: Expression, field: Expression) -> Expression:
        """Return the functional derivative of the integral of a density by a field.

        The derivative is with respect to the field.
        """
        derivatives: dict[int, Expression] = {}
        curvature = None
        for item in self._walk(density):
            kind = item.operator()
            if kind == partial and item.operands()[0].is_trivially_equal(field):
                derivatives[int(item.operands()[1])] = item
            elif kind == laplacian and item.operands()[0].is_trivially_equal(field):
                curvature = item
        slots = {item: SR.var(f'_slot_{axis}') for axis, item in derivatives.items()}
        if curvature is not None:
            slots[curvature] = SR.var('_slot_laplacian')
        flat = density.subs(slots)
        back = {symbol: item for item, symbol in slots.items()}
        variation = flat.diff(field).subs(back)
        fluxes = {axis: flat.diff(slots[item]).subs(back) for axis, item in derivatives.items()}
        if fluxes:
            variation -= self._get_flux_divergence(fluxes, derivatives, field)
        if curvature is not None:
            variation += self._get_laplacian(flat.diff(slots[curvature]).subs(back), field)
        return variation

    def _get_laplacian(self, expression: Expression, field: Expression) -> Expression:
        """Return the Laplacian of an expression of a field, term by term.

        The factors free of the field stay outside. The linear terms of the field, such as `laplacian(x)`, stay apart,
        so a spectral discretization finds them.
        """
        expanded = SR(expression).expand()
        terms = expanded.operands() if expanded.operator() is add_vararg else [expanded]
        total = SR(0)
        for term in terms:
            factors = term.operands() if term.operator() is mul_vararg else [term]
            constant = prod((factor for factor in factors if not factor.has(field)), start=SR(1))
            varying = prod((factor for factor in factors if factor.has(field)), start=SR(1))
            if varying.has(field):
                total += constant * laplacian(varying)
        return total

    def _get_matrix_derivative(self, potential: Expression, state: Matrix) -> Matrix:
        """Return the matrix gradient of a potential at a tensor state.

        It is written in the matrices of the symmetry of the state. The derivative with respect to the independent
        components is the product of the matrices of the components, the metric of the state, and the coefficients of
        the gradient, so the metric gives the coefficients.
        """
        if self.conserved:
            error_msg = f'{self.label!r} conserves the tensor state {state}. Give a scalar state.'
            raise ValidationError(error_msg)
        components = sorted(state.variables(), key=str)
        units = [state.derivative(component) for component in components]
        metric = matrix(SR, [[(first.transpose() * second).trace() for second in units] for first in units])
        density = distribute_operators(potential)
        derivatives = [
            self._get_variation(density, component) if self.variational else density.diff(component)
            for component in components
        ]
        coefficients = metric.inverse() * vector(SR, derivatives)
        total = matrix(SR, state.nrows(), state.ncols())
        for coefficient, unit in zip(coefficients, units, strict=True):
            total += coefficient * unit
        return total

    def _get_derivative(self, potential: Expression, state: Target | Matrix) -> Target | Matrix:
        """Return the derivative of the potential with respect to a state.

        For a vector state, it returns that of every component.
        """
        if isinstance(state, Matrix):
            return self._get_matrix_derivative(potential, state)
        if isinstance(state, FreeModuleElement):
            if self.variational or self.conserved:
                error_msg = f'{self.label!r} moves the vector state {state} of a field. Give a scalar state.'
                raise ValidationError(error_msg)
            return vector(SR, [potential.diff(component) for component in state])
        if self.variational:
            return self._get_variation(potential, state)
        return potential.diff(state)

    def _get_members(self, namespace: SimpleNamespace) -> list[SimpleNamespace]:
        """Return the namespaces of the members whose states move.

        These are the members of every role that binds one path.
        """
        return [
            getattr(namespace, name)
            for name, spec in self._get_specs().items()
            if isinstance(spec, RoleSpec) and not spec.variadic and not spec.noise
        ]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseComponent]:
        """Return the rate of every moving state.

        The rate is minus the derivative of the potential over the divisor.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the roles.

        Returns:
            laws:
                The rate laws of the moving states.
        """
        potential = SR(self._get_potential(namespace))
        laws: list[BaseComponent] = []
        for member in self._get_members(namespace):
            for name in self._get_names():
                state = getattr(member, name)
                derivative = self._get_derivative(potential, state)
                rate = self._get_laplacian(derivative, state) if self.conserved else -derivative
                divisor = self._get_divisor(member, name)
                if divisor is not None:
                    rate = rate / divisor
                laws.append(RateLaw(self._get_rate_state(member, name), rate))
        return laws
