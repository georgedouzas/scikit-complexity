"""Define the interaction that gives the equations of motion of a Lagrangian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression, function
from sage.modules.free_module_element import FreeModuleElement

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, ConstraintLaw
from ....core.errors import ValidationError
from ....core.types import Role


class EulerLagrangeInteraction(Interaction):
    """An interaction that moves coordinates by the equations of a Lagrangian.

    A Lagrangian `L` is the kinetic minus the potential energy of a mechanical system. The interaction gives the
    equations of motion `d/dt dL/dq' - dL/dq = Q - dR/dq'` for the coordinates `q`, with the dissipation function `R`
    and the generalized forces `Q`. It returns one equation per coordinate, and the compiler solves them together for
    the accelerations, so a Lagrangian with a mass matrix, such as that of a double pendulum, needs no inversion by
    hand. Equations of one coordinate from several interactions add, so a spring between two
    pendulums is an interaction of its own whose Lagrangian holds only the coupling energy.

    A Lagrangian, dissipation, or forcing function takes a namespace with the roles, the time `t`, and the parameters a
    subclass declares, and writes the velocity of a coordinate `q` as `q.diff(t)`. The coordinates are the states of
    every role that binds one element or group, `target` by default, and every component of a vector state is a
    coordinate. Every other value of the roles, such as a mass or an inertia, is held fixed when the momenta change in
    time, so it stays a parameter.

    Unlike `HamiltonianInteraction`, which needs canonical momenta, it takes the energy in coordinates and velocities.
    Choose it when the energy is easier to write than the forces, as for linked bodies.

    It appears in many fields. Pendulums and linkages in mechanics, electric circuits written by their energy, and the
    joints of a robot arm follow the equations of a Lagrangian.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'pendulum'`.

        target:
            The path of the element or group whose states are the coordinates.

        states:
            The name of the state that is a coordinate, or the names of several states.

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
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.mechanics import EulerLagrangeInteraction
        >>> swing = EulerLagrangeInteraction(
        ...     'swing',
        ...     target='bob',
        ...     states='q',
        ...     lagrangian=lambda s: s.target.q.diff(s.t) ** 2 / 2 + cos(s.target.q),
        ... )
        >>> system = System(elements=[Element('bob', values={'q': 0.1})], interactions=[swing])
        >>> trajectory = system.prepare().simulate(until=3.2, dt=0.01)
        >>> round(float(trajectory.get_values('bob.q')[-1]), 2)
        -0.1
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
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

    def _check_arguments(self) -> None:
        """Raise an error for a state that is not a name, or an uncallable function.

        Raises:
            ValidationError:
                The type declares states, a state is not a name, or a function is not callable.
        """
        super()._check_arguments()
        names = self._get_names()
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'{type(self).__name__} {self.label!r} reads the states {self.states!r}. Give names of states.'
            raise ValidationError(error_msg)
        for name in ('lagrangian', 'dissipation', 'forcing'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace.'
                raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the laws of a binding.

        Args:
            context:
                The namespace of the elements it binds, whose Euler-Lagrange equations it sets in `laws_`.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = self._get_equations(context)

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
        for name in self._get_roles(single=True):
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

    def _get_lagrangian_of(self, namespace: SimpleNamespace, velocities: Mapping[Expression, Expression]) -> Expression:
        """Return the Lagrangian in the symbols of the states and of the velocities.

        The namespace holds the states as symbols, and the velocity `q.diff(t)` of a coordinate `q` becomes the symbol
        `velocities[q]`, as the invariants analysis reads it.
        """
        time = getattr(namespace, 't', SR.var('t'))
        functions: dict[str, Expression] = {}
        functional = SimpleNamespace(
            **{
                name: self._get_functional(item, time, functions) if isinstance(item, SimpleNamespace | tuple) else item
                for name, item in vars(namespace).items()
            },
        )
        rates = {function(str(q))(time).diff(time): velocity for q, velocity in velocities.items()}
        states = {function(name)(time): symbol for name, symbol in functions.items()}
        return SR(self._get_lagrangian(functional)).subs(rates).subs(states)

    @staticmethod
    def _get_values(value: object) -> list[Expression]:
        """Return the values of a namespace of roles, with every component of a vector."""
        if isinstance(value, SimpleNamespace):
            return [item for member in vars(value).values() for item in EulerLagrangeInteraction._get_values(member)]
        if isinstance(value, tuple | FreeModuleElement):
            return [item for member in value for item in EulerLagrangeInteraction._get_values(member)]
        return [value] if isinstance(value, Expression) else []

    def _get_held(self, namespace: SimpleNamespace, coordinates: Sequence[Expression]) -> dict[Expression, Expression]:
        """Return a symbol for every value of the roles that is a function of time.

        The coordinates are left out. Such a value, such as the inertia `J(t)` of a rotor, is held fixed while a
        momentum changes in time.
        """
        time = SR(namespace.t)
        values = [
            item
            for name in self._get_roles(single=True)
            for item in self._get_values(getattr(namespace, name))
            if item.operator() is not None and item.operands() == [time] and item not in coordinates
        ]
        return {item: SR.var(f'_held_{index}') for index, item in enumerate(dict.fromkeys(values))}

    def _get_equations(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the Euler-Lagrange equation of every coordinate.

        The coordinates, their velocities `q.diff(t)`, and their accelerations `q.diff(t, 2)` take symbols while the
        Lagrangian is differentiated, and the equations are written back in them. The other values of the roles are
        held fixed while a momentum changes in time.
        """
        time = SR(namespace.t)
        coordinates = self._get_coordinates(namespace)
        positions = [SR.var(f'_position_{index}') for index in range(len(coordinates))]
        slots = [SR.var(f'_velocity_{index}') for index in range(len(coordinates))]
        accelerations = [SR.var(f'_acceleration_{index}') for index in range(len(coordinates))]
        rates = {q.diff(time): slot for q, slot in zip(coordinates, slots, strict=True)}
        states = dict(zip(coordinates, positions, strict=True))
        held = self._get_held(namespace, coordinates)
        released = {symbol: value for value, symbol in held.items()}

        def slotted(value: object) -> Expression:
            return SR(value).subs(rates).subs(states)

        back = {
            **{slot: q.diff(time) for q, slot in zip(coordinates, slots, strict=True)},
            **{item: q for q, item in zip(coordinates, positions, strict=True)},
            **{item: q.diff(time, 2) for q, item in zip(coordinates, accelerations, strict=True)},
        }
        lagrangian = slotted(self._get_lagrangian(namespace))
        dissipation = slotted(0 if self.dissipation is None else self.dissipation(namespace))
        forces = [slotted(force) for force in self._get_forces(namespace, len(coordinates))]
        laws: list[BaseLaw] = []
        for q, position, slot, force in zip(coordinates, positions, slots, forces, strict=True):
            momentum = lagrangian.diff(slot)
            change = momentum.subs(held).diff(time).subs(released) + sum(
                momentum.diff(other) * other_slot + momentum.diff(other_slot) * acceleration
                for other, other_slot, acceleration in zip(positions, slots, accelerations, strict=True)
            )
            lhs = (change - lagrangian.diff(position)).subs(back)
            rhs = (force - dissipation.diff(slot)).subs(back)
            laws.append(ConstraintLaw(lhs == rhs, target=q))
        return laws
