"""Define Hamiltonian interactions, Hamilton's equations of canonical coordinates and momenta."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import SR, Expression, vector
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, RateLaw
from ..types import Target
from ._interaction import Interaction


class HamiltonianInteraction(Interaction):
    """Hamilton's equations `q' = dH/dp` and `p' = -dH/dq` of the canonical pairs `(q, p)` of a Hamiltonian `H(ns)`.

    The coordinates and the momenta are states of order 1, which may be angles, of every role that binds one element or
    group, `target` by default, and the coordinates pair with the momenta in order. Every component of a vector state
    is a coordinate or a momentum. The interaction returns one rate law per coordinate and per momentum, and the rates
    of one state add, so the Hamiltonians of separate interactions sum: a coupling between two oscillators is an
    interaction of its own whose Hamiltonian holds only the coupling energy. Dissipation or forcing is a rate
    interaction on the momenta.

    A Hamiltonian function takes the namespace of the interaction, with the namespaces of the roles, the time `t`, and
    the symbols of the parameters a subclass declares.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose states are the canonical pairs.

        coordinates:
            The name of the state of order 1 that is a coordinate, or the names of several states.

        momenta:
            The name of the state of order 1 that is the momentum of the coordinate, or the names of several states in
            the order of the coordinates.

        hamiltonian:
            A function of the namespace that returns the Hamiltonian.

    Raises:
        ValidationError:
            A state is not a name, the coordinates and the momenta differ in count, the Hamiltonian is not callable,
            or a coordinate and its momentum differ in size or are not of order 1.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import HamiltonianInteraction
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class OscillatorElement(Element):
        ...     def __init__(self, label, q: Annotated[float, StateSpec()] = 1.0, p: Annotated[float, StateSpec()] = 0):
        ...         super().__init__(label)
        ...         self.q = q
        ...         self.p = p
        >>> energy = lambda s: (s.target.p**2 + s.target.q**2) / 2
        >>> spring = HamiltonianInteraction('spring', target='mass', hamiltonian=energy)
        >>> system = System(elements=[OscillatorElement('mass')], interactions=[spring])
        >>> trajectory = simulate(system, until=3.14159, dt=0.01)
        >>> round(float(trajectory.values('mass.q')[-1]), 3)
        -1.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('coordinates', 'momenta'))],
        coordinates: str | Sequence[str] = 'q',
        momenta: str | Sequence[str] = 'p',
        hamiltonian: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.coordinates = coordinates
        self.momenta = momenta
        self.hamiltonian = hamiltonian

    def _get_names(self, names: str | Sequence[str]) -> tuple[str, ...]:
        """Return the names of the states of one side of the canonical pairs."""
        return (names,) if isinstance(names, str) else tuple(names)

    def _check(self) -> None:
        """Raise an error for a state that is not a name, an unpaired state, or a Hamiltonian that is not callable.

        Raises:
            ValidationError:
                The type declares states, a state is not a name, the coordinates and the momenta differ in count, or
                the Hamiltonian is not callable.
        """
        super()._check()
        for side in ('coordinates', 'momenta'):
            names = self._get_names(getattr(self, side))
            if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
                error_msg = f'The {side} of {self.label!r} are {getattr(self, side)!r}. Give names of states.'
                raise ValidationError(error_msg)
        coordinates, momenta = self._get_names(self.coordinates), self._get_names(self.momenta)
        if len(coordinates) != len(momenta):
            unpaired = coordinates[len(momenta) :] or momenta[len(coordinates) :]
            error_msg = (
                f'{type(self).__name__} {self.label!r} pairs the coordinates {coordinates} with the momenta {momenta}, '
                f'so {", ".join(unpaired)} has no partner. Give one momentum per coordinate.'
            )
            raise ValidationError(error_msg)
        if self.hamiltonian is not None and not callable(self.hamiltonian):
            error_msg = f'The hamiltonian of {self.label!r} is {self.hamiltonian!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _get_hamiltonian(self, namespace: SimpleNamespace) -> object:
        """Return the Hamiltonian, the hook a subclass overrides to derive it from its parameters."""
        if self.hamiltonian is None:
            error_msg = f'{type(self).__name__} {self.label!r} has no Hamiltonian. Give a function of the namespace.'
            raise ValidationError(error_msg)
        return self.hamiltonian(namespace)

    def _get_members(self, namespace: SimpleNamespace) -> list[SimpleNamespace]:
        """Return the namespaces of the members whose states are canonical pairs, those of every role of one path."""
        return [
            getattr(namespace, name)
            for name, spec in self._get_specs().items()
            if isinstance(spec, RoleSpec) and not spec.variadic and not spec.noise
        ]

    def _get_state(self, member: SimpleNamespace, name: str) -> Target:
        """Return a canonical state of a member, after checking that it has order 1."""
        if hasattr(member, f'{name}_d1'):
            error_msg = (
                f'{type(self).__name__} {self.label!r} reads the state {name} of order above 1, but canonical '
                'coordinates and momenta have order 1. Declare it with `StateSpec(order=1)`, or derive the equations '
                'of a coordinate of order 2 with `EulerLagrangeInteraction`.'
            )
            raise ValidationError(error_msg)
        state: Target = getattr(member, name)
        return state

    def _get_pairs(self, namespace: SimpleNamespace) -> list[tuple[Target, Target]]:
        """Return the canonical pairs of every member, after checking that each coordinate matches its momentum."""
        pairs = []
        names = list(zip(self._get_names(self.coordinates), self._get_names(self.momenta), strict=True))
        for member in self._get_members(namespace):
            for coordinate, momentum in names:
                q, p = self._get_state(member, coordinate), self._get_state(member, momentum)
                if len(self._get_components(q)) != len(self._get_components(p)):
                    error_msg = (
                        f'The coordinate {coordinate} and the momentum {momentum} of {self.label!r} differ in size. '
                        'Give a momentum of the size of its coordinate.'
                    )
                    raise ValidationError(error_msg)
                pairs.append((q, p))
        return pairs

    def _get_components(self, state: Target) -> list[Expression]:
        """Return the components of a state, the state itself for a scalar."""
        return list(state) if isinstance(state, FreeModuleElement) else [state]

    def _get_derivative(self, hamiltonian: Expression, state: Target) -> Target:
        """Return the derivative of the Hamiltonian with respect to a state, every component of a vector state."""
        if isinstance(state, FreeModuleElement):
            return vector(SR, [hamiltonian.diff(component) for component in state])
        return hamiltonian.diff(state)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the rate of every coordinate, `dH/dp`, and of every momentum, `-dH/dq`.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the roles.

        Returns:
            laws:
                The rate laws of the coordinates and the momenta.
        """
        pairs = self._get_pairs(namespace)
        hamiltonian = SR(self._get_hamiltonian(namespace))
        laws: list[BasePart] = []
        for q, p in pairs:
            laws.append(RateLaw(q, self._get_derivative(hamiltonian, p)))
            laws.append(RateLaw(p, -self._get_derivative(hamiltonian, q)))
        return laws
