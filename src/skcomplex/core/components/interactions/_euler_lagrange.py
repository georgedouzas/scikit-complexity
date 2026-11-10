"""Define the interaction that gives the equations of motion of a Lagrangian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, EquationLaw
from ..laws.utils import derivative
from ..spaces import BaseSpace
from ._interaction import Interaction


class EulerLagrangeInteraction(Interaction):
    """An interaction that moves coordinates by the equations of a Lagrangian.

    A Lagrangian `L` is the kinetic minus the potential energy of a mechanical system. The interaction gives the
    equations of motion `d/dt dL/dq' - dL/dq = Q - dR/dq'` for the coordinates `q`, states of order 2, with the
    dissipation function `R` and the generalized forces `Q`. It returns one equation per coordinate, and the compiler
    solves them together for the accelerations, so a Lagrangian with a mass matrix, such as that of a double pendulum,
    needs no inversion by hand. Equations of one coordinate from several interactions add, so a spring between two
    pendulums is an interaction of its own whose Lagrangian holds only the coupling energy.

    A Lagrangian, dissipation, or forcing function takes a namespace with the roles, the time `t`, and the parameters a
    subclass declares. The coordinates are the states of every role that binds one element or group, `target` by
    default, and every component of a vector state is a coordinate.

    Unlike `HamiltonianInteraction`, which needs canonical momenta, it takes the energy in coordinates and velocities.
    Choose it when the energy is easier to write than the forces, as for linked bodies.

    It appears in many fields. Pendulums and linkages in mechanics, electric circuits written by their energy, and the
    joints of a robot arm follow the equations of a Lagrangian.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'pendulum'`.

        target:
            The path of the element or group whose states are the coordinates.

        states:
            The name of the state of order 2 that is a coordinate, or the names of several states.

        lagrangian:
            A function of the namespace that returns the Lagrangian.

        dissipation:
            A function of the namespace that returns the Rayleigh dissipation function, or `None` for none.

        forcing:
            A function of the namespace that returns the generalized forces, or `None` for none.

    Raises:
        ValidationError:
            A state is not a name, or a function is not callable.

    Examples:
        >>> from typing import Annotated
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import EulerLagrangeInteraction
        >>> from skcomplex.core.components.laws.utils import derivative
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class AngleElement(Element):
        ...     def __init__(self, label, q: Annotated[float, StateSpec(order=2)] = 0.1, q_d1: float = 0.0):
        ...         super().__init__(label)
        ...         self.q = q
        ...         self.q_d1 = q_d1
        >>> swing = EulerLagrangeInteraction(
        ...     'swing',
        ...     target='bob',
        ...     states='q',
        ...     lagrangian=lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q),
        ... )
        >>> system = CompositeSystem(elements=[AngleElement('bob')], interactions=[swing])
        >>> trajectory = simulate(system, until=3.2, dt=0.01)
        >>> round(float(trajectory.get_values('bob.q')[-1]), 2)
        -0.1
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('states',))],
        states: str | Sequence[str] = 'x',
        lagrangian: Callable[[SimpleNamespace], object] | None = None,
        dissipation: Callable[[SimpleNamespace], object] | None = None,
        forcing: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.states = states
        self.lagrangian = lagrangian
        self.dissipation = dissipation
        self.forcing = forcing

    def _get_names(self) -> tuple[str, ...]:
        """Return the names of the states that are coordinates."""
        return (self.states,) if isinstance(self.states, str) else tuple(self.states)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a state that is not a name, or an uncallable function.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, a state is not a name, or a function is not callable.
        """
        super()._prepare(context)
        names = self._get_names()
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'{type(self).__name__} {self.label!r} reads the states {self.states!r}. Give names of states.'
            raise ValidationError(error_msg)
        for name in ('lagrangian', 'dissipation', 'forcing'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace.'
                raise ValidationError(error_msg)

    def _get_lagrangian(self, namespace: SimpleNamespace) -> object:
        """Return the Lagrangian.

        This is the hook a subclass overrides to derive it from its parameters.
        """
        if self.lagrangian is None:
            error_msg = f'{type(self).__name__} {self.label!r} has no Lagrangian. Give a function of the namespace.'
            raise ValidationError(error_msg)
        return self.lagrangian(namespace)

    def _get_coordinates(self, namespace: SimpleNamespace) -> list[Expression]:
        """Return the coordinates.

        They are every component of the states of every role that binds one path.
        """
        coordinates = []
        for name, spec in self._get_specs().items():
            if not isinstance(spec, RoleSpec) or spec.variadic or spec.noise:
                continue
            for state in self._get_names():
                value = getattr(getattr(namespace, name), state)
                coordinates += list(value) if isinstance(value, FreeModuleElement) else [value]
        return coordinates

    def _get_forces(self, namespace: SimpleNamespace, count: int) -> list[Expression]:
        """Return the generalized force of every coordinate, zero without a forcing."""
        if self.forcing is None:
            return [SR(0)] * count
        forces = self.forcing(namespace)
        forces = list(forces) if isinstance(forces, FreeModuleElement | list | tuple) else [forces]
        if len(forces) != count:
            error_msg = f'The forcing of {self.label!r} returns {len(forces)} forces for {count} coordinates.'
            raise ValidationError(error_msg)
        return [SR(force) for force in forces]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the Euler-Lagrange equation of every coordinate.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the roles.

        Returns:
            laws:
                The equation laws of the coordinates.
        """
        coordinates = self._get_coordinates(namespace)
        velocities = {derivative(q, 1): SR.var(f'_velocity_{index}') for index, q in enumerate(coordinates)}
        back = {symbol: item for item, symbol in velocities.items()}
        slots = list(velocities.values())
        lagrangian = SR(self._get_lagrangian(namespace)).subs(velocities)
        dissipation = SR(0 if self.dissipation is None else self.dissipation(namespace)).subs(velocities)
        forces = self._get_forces(namespace, len(coordinates))
        time = SR(namespace.t)
        laws: list[BasePart] = []
        for q, slot, force in zip(coordinates, slots, forces, strict=True):
            momentum = lagrangian.diff(slot)
            change = momentum.diff(time) + sum(
                momentum.diff(other) * other_slot + momentum.diff(other_slot) * derivative(other, 2)
                for other, other_slot in zip(coordinates, slots, strict=True)
            )
            lhs = (change - lagrangian.diff(q)).subs(back)
            rhs = (force - dissipation.diff(slot)).subs(back)
            laws.append(EquationLaw(lhs == rhs, target=q))
        return laws
