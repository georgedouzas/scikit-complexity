"""Define the interaction that gives Hamilton's equations of a Hamiltonian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression, vector
from sage.modules.free_module_element import FreeModuleElement

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.components.types import Target
from .....core.errors import ValidationError
from .....core.specs import Role


class HamiltonianInteraction(Interaction):
    """An interaction that moves canonical pairs by Hamilton's equations.

    A Hamiltonian `H` is the total energy of a system as a function of its coordinates `q` and momenta `p`. The
    interaction gives Hamilton's equations `q' = dH/dp` and `p' = -dH/dq`, one rate per coordinate and per momentum. The
    coordinates and momenta are states of order 1 of every role that binds one element or group, `target` by default,
    and they pair in order. The rates of one state add, so a coupling between two oscillators is an interaction of its
    own whose Hamiltonian holds only the coupling energy. Dissipation or forcing is a `RateInteraction` on the momenta.

    Unlike `EulerLagrangeInteraction`, which takes coordinates and velocities, it takes coordinates and momenta and
    gives first order rates, which symplectic solvers keep on the surface of constant energy. Choose it when the energy
    is known in canonical form and long runs must conserve it.

    It appears in many fields. Planets orbit a star, atoms vibrate in a molecule, and charged particles move in a
    magnetic trap by Hamilton's equations.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'orbit'`.

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
            or a coordinate and its momentum differ in size. Preparing a system raises it too when another law gives
            a coordinate a rate of a higher derivative.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.mechanics import HamiltonianInteraction
        >>> energy = lambda s: (s.target.p**2 + s.target.q**2) / 2
        >>> spring = HamiltonianInteraction('spring', target='mass', hamiltonian=energy)
        >>> system = System(elements=[Element('mass', values={'q': 1.0, 'p': 0.0})], interactions=[spring]).prepare()
        >>> trajectory = system.simulate(until=3.14159, dt=0.01)
        >>> round(float(trajectory.get_values('mass.q')[-1]), 3)
        -1.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the canonical pairs, after checking the states and the Hamiltonian.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of its roles, the time `t`, and
                the parameters.

        Raises:
            ValidationError:
                The type declares states, a state is not a name, the coordinates and the momenta differ in count, or
                the Hamiltonian is not callable.
        """
        super()._prepare(context)
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
        if isinstance(context, SimpleNamespace):
            self.laws_ = self._get_canonical_laws(context)

    def _get_hamiltonian(self, namespace: SimpleNamespace) -> object:
        """Return the Hamiltonian.

        This is the hook a subclass overrides to derive it from its parameters.
        """
        if self.hamiltonian is None:
            error_msg = f'{type(self).__name__} {self.label!r} has no Hamiltonian. Give a function of the namespace.'
            raise ValidationError(error_msg)
        return self.hamiltonian(namespace)

    def _get_members(self, namespace: SimpleNamespace) -> list[SimpleNamespace]:
        """Return the namespaces of the members whose states are canonical pairs.

        These are the members of every role of one path.
        """
        return [getattr(namespace, name) for name in self._get_roles(single=True)]

    def _get_pairs(self, namespace: SimpleNamespace) -> list[tuple[Target, Target]]:
        """Return the canonical pairs of every member.

        It checks that each coordinate matches its momentum.
        """
        pairs = []
        names = list(zip(self._get_names(self.coordinates), self._get_names(self.momenta), strict=True))
        for member in self._get_members(namespace):
            for coordinate, momentum in names:
                q, p = getattr(member, coordinate), getattr(member, momentum)
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
        """Return the derivative of the Hamiltonian with respect to a state.

        For a vector state, it returns that of every component.
        """
        if isinstance(state, FreeModuleElement):
            return vector(SR, [self._get_partial(hamiltonian, component) for component in state])
        return self._get_partial(hamiltonian, state)

    @staticmethod
    def _get_partial(hamiltonian: Expression, state: Expression) -> Expression:
        """Return the partial derivative of the Hamiltonian with respect to a scalar state.

        A state that is a function of the time, such as `q(t)`, stands in for a symbol while the derivative is taken.
        """
        if state.is_symbol():
            return hamiltonian.diff(state)
        symbol = SR.symbol()
        return hamiltonian.subs({state: symbol}).diff(symbol).subs({symbol: state})

    def _get_canonical_laws(self, namespace: SimpleNamespace) -> list[BaseComponent]:
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
        laws: list[BaseComponent] = []
        for q, p in pairs:
            laws.append(RateLaw(q, self._get_derivative(hamiltonian, p)))
            laws.append(RateLaw(p, -self._get_derivative(hamiltonian, q)))
        return laws
