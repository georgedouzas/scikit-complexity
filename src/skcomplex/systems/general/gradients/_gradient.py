"""Define the interaction that moves states down the gradient of a potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterator
from math import prod
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression, matrix, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.symbolic.operators import add_vararg, mul_vararg

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import distribute_operators, laplacian, partial
from ....core.components.types import Target
from ....core.errors import ValidationError
from ....core.types import Role


class GradientInteraction(Interaction):
    """An interaction that moves states down the gradient of a potential.

    Every state `x` that the interaction moves follows the gradient flow `x' = -dU/dx`, where `U` is the potential. A
    vector state gets the gradient of every component. With `inertia` the force moves the second derivative instead,
    Newton's law `RateLaw(x.diff(t, 2), -dU/dx / m)`, divided by the mass `m` of the member when it has one.

    On a field, the potential is the density `f` of a functional `F`, the integral of `f` over space, and may read the
    field, its derivatives, and its Laplacian. With `variational` the field moves by the functional derivative of `F`,
    as in the Allen-Cahn equation. With `conserved` the rate is the Laplacian of that derivative, as in the
    Cahn-Hilliard equation, so the total of the field stays constant. The potential function takes a namespace with the
    roles, the `sources`, the time `t`, and the parameters a subclass declares.

    Unlike a `RateLaw`, which takes the rate itself, it takes a potential and derives the rates from it. Choose it
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

        variational:
            Whether the potential is the density of a functional of a field, moved by its functional derivative.

        conserved:
            Whether the rate is the Laplacian of the derivative of the potential, which conserves the total of a field.

        inertia:
            Whether the force moves the second derivative of every state, divided by the mass `m` of the member when it
            has one.

    Raises:
        ValidationError:
            A state is not a name, or the potential is not callable.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.gradients import GradientInteraction
        >>> well = GradientInteraction('well', target='walker', states='x', potential=lambda s: s.target.x**2 / 2)
        >>> system = System(elements=[Element('walker', values={'x': 1.0})], interactions=[well])
        >>> trajectory = system.prepare().simulate(until=1.0, dt=0.01)
        >>> round(float(trajectory.get_values('walker.x')[-1]), 4)
        0.3679
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        states: str | tuple[str, ...] = 'x',
        potential: Callable[[SimpleNamespace], object] | None = None,
        sources: tuple[Role, ...] = (),
        variational: bool = False,
        conserved: bool = False,
        inertia: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.states = states
        self.potential = potential
        self.sources = sources
        self.variational = variational
        self.conserved = conserved
        self.inertia = inertia

    def _get_names(self) -> tuple[str, ...]:
        """Return the names of the states the interaction moves."""
        return (self.states,) if isinstance(self.states, str) else tuple(self.states)

    def _check_arguments(self) -> None:
        """Raise an error for a state that is not a name, or a bad potential.

        Raises:
            ValidationError:
                A state is not a name, or the potential is not callable.
        """
        super()._check_arguments()
        names = self._get_names()
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'{type(self).__name__} {self.label!r} moves the states {self.states!r}. Give names of states.'
            raise ValidationError(error_msg)
        if self.potential is not None and not callable(self.potential):
            error_msg = f'The potential of {self.label!r} is {self.potential!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the laws of a binding.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        time = getattr(context, 't', SR.var('t'))
        namespace = self._get_symbolic(context, time)
        self.laws_: list[BaseComponent] = self._get_gradient_laws(namespace, context, time)

    def _get_potential(self, namespace: SimpleNamespace) -> object:
        """Return the potential.

        This is the hook a subclass overrides to derive it from its parameters.
        """
        if self.potential is None:
            error_msg = f'{type(self).__name__} {self.label!r} has no potential. Give a function of the namespace.'
            raise ValidationError(error_msg)
        return self.potential(namespace)

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
        return [getattr(namespace, name) for name in self._get_roles(single=True)]

    def _get_gradient_laws(
        self,
        namespace: SimpleNamespace,
        context: SimpleNamespace,
        time: Expression,
    ) -> list[BaseComponent]:
        """Return the rate of every moving state, down the potential.

        The potential is differentiated in the symbols of `namespace`, and the laws target the states of `context`, the
        same states as functions of the time.
        """
        potential = SR(self._get_potential(namespace))
        laws: list[BaseComponent] = []
        members = zip(self._get_members(namespace), self._get_members(context), strict=True)
        for member, functional in members:
            for name in self._get_names():
                state = getattr(member, name)
                derivative = self._get_derivative(potential, state)
                rate = self._get_laplacian(derivative, state) if self.conserved else -derivative
                target = getattr(functional, name)
                if not self.inertia:
                    laws.append(RateLaw(target, rate))
                    continue
                mass = getattr(member, 'm', None)
                laws.append(RateLaw(target.diff(time, 2), rate if mass is None else rate / mass))
        return laws
