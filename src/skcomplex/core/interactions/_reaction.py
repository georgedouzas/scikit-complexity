"""Define reactions, jumps that move units between states by mass action or a given propensity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, NoiseLaw, RateLaw
from ..laws.utils import where
from ._jump import JumpInteraction

MODES = ('rate', 'jump', 'langevin')
ENTRY_SIZE = 2

type Entries = Mapping[tuple[str, str], float]


class ReactionInteraction(JumpInteraction):
    """Reaction that moves units from its reactants to its products at a mass-action or a given propensity.

    A reactant, a product, or a modifier is a pair of the path of an element, or of a member of a group such as
    `cells[0]`, and the name of its state that holds the units, mapped to its stoichiometric coefficient, a real number.
    So `S + I -> 2 I` has the reactants `{('S', 'n'): 1, ('I', 'n'): 1}` and the products `{('I', 'n'): 2}`, and two
    states of one element are `('cell', 'u')` and `('cell', 'v')`. The propensity is `k` times the product of every
    reactant and modifier state to the power of its coefficient, or the expression the `propensity` function returns.
    The reaction changes every state by its coefficient among the products minus its coefficient among the reactants,
    so it conserves the total of the states whose changes sum to zero, and modifiers enter the propensity but do not
    change.

    In the mode `'rate'` the reaction is a rate equation. In the mode `'jump'` it is the random event of a
    `JumpInteraction` at the propensity that moves the units of the coefficients. In the mode `'langevin'` it is the
    chemical Langevin equation, the rate equation with one Wiener process of amplitude the square root of the propensity
    driving every state the reaction changes.

    Args:
        label:
            The label of the reaction, unique among the parts of its system.

        reactants:
            The coefficients of the reactants, or `None` for a source.

        products:
            The coefficients of the products, or `None` for a sink.

        k:
            The rate constant.

        propensity:
            A function of the namespace of the reaction that returns the propensity, or `None` for mass action. The
            namespace holds the tuples `reactants`, `products`, and `modifiers` of the namespaces of the bound elements,
            in the order of their entries, and the symbol `k`.

        modifiers:
            The coefficients of the modifiers, whose states the propensity reads but the reaction does not change, or
            `None`.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

    Raises:
        ValidationError:
            The mode is unknown, the reaction has neither reactants nor products, an entry is not a pair of a path and a
            state name, or a coefficient is not a positive number.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.interactions import ReactionInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> infection = ReactionInteraction(
        ...     'infection', reactants={('S', 'n'): 1, ('I', 'n'): 1}, products={('I', 'n'): 2}, k=0.0005
        ... )
        >>> recovery = ReactionInteraction('recovery', reactants={('I', 'n'): 1}, products={('R', 'n'): 1}, k=0.1)
        >>> system = System(
        ...     elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        ...     interactions=[infection, recovery],
        ... )
        >>> states = Simulation(system, dt=1.0).check().prepare().run(until=50).trajectory_.states
        >>> round(float(states[-1].sum()), 6)
        1000.0
    """

    def __init__(
        self,
        label: str,
        *,
        reactants: Annotated[Entries | None, RoleSpec(variadic=True)] = None,
        products: Annotated[Entries | None, RoleSpec(variadic=True)] = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        propensity: Callable[[SimpleNamespace], Expression] | None = None,
        modifiers: Annotated[Entries | None, RoleSpec(variadic=True)] = None,
        mode: str = 'rate',
    ) -> None:
        super().__init__(label, target='')
        self.reactants = reactants
        self.products = products
        self.k = k
        self.propensity = propensity
        self.modifiers = modifiers
        self.mode = mode

    def _get_terms(
        self,
        entries: Entries | None,
        namespaces: Sequence[SimpleNamespace],
    ) -> list[tuple[Expression, float]]:
        """Return the state symbol and the coefficient of every entry, read from the namespaces of its elements."""
        items = list((entries or {}).items())
        return [
            (getattr(namespace, state), coefficient)
            for ((_, state), coefficient), namespace in zip(items, namespaces, strict=True)
        ]

    def _get_changes(self, namespace: SimpleNamespace) -> dict[Expression, float]:
        """Return the net change of every state the reaction changes, products minus reactants."""
        changes: dict[Expression, float] = {}
        for sign, entries, namespaces in (
            (-1, self.reactants, namespace.reactants),
            (1, self.products, namespace.products),
        ):
            for state, coefficient in self._get_terms(entries, namespaces):
                changes[state] = changes.get(state, 0) + sign * coefficient
        return {state: change for state, change in changes.items() if change}

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the propensity of the reaction, by the user's function or by mass action."""
        if self.propensity is not None:
            return SR(self.propensity(namespace))
        propensity = SR(namespace.k)
        for entries, namespaces in ((self.reactants, namespace.reactants), (self.modifiers, namespace.modifiers)):
            for state, coefficient in self._get_terms(entries, namespaces):
                propensity *= state**coefficient
        return propensity

    def _get_langevin_laws(self, propensity: Expression, changes: dict[Expression, float]) -> list[BaseLaw]:
        """Return the drift of the reaction and the noise of its one Wiener process on every state it changes."""
        amplitude = sqrt(where(propensity > 0, propensity, 0))
        rates: list[BaseLaw] = [RateLaw(state, change * propensity) for state, change in changes.items()]
        noises: list[BaseLaw] = [
            NoiseLaw(state, change * amplitude, process=self.label) for state, change in changes.items()
        ]
        return rates + noises

    def _check_entries(self, name: str) -> None:
        """Raise an error for entries that are not pairs of a path and a state name with positive coefficients."""
        entries = getattr(self, name)
        if entries is None:
            return
        valid = isinstance(entries, Mapping) and all(
            isinstance(entry, tuple)
            and len(entry) == ENTRY_SIZE
            and all(isinstance(item, str) for item in entry)
            and entry[1].isidentifier()
            and isinstance(coefficient, int | float)
            and coefficient > 0
            for entry, coefficient in entries.items()
        )
        if not valid:
            error_msg = (
                f'The {name} of reaction {self.label!r} are {entries!r}. Map pairs of a path and a state name to '
                'positive coefficients, such as {("S", "n"): 1}.'
            )
            raise ValidationError(error_msg)

    def _check_jump(self) -> None:
        """Raise an error for an unknown mode, no reactants and products, or entries the reaction cannot take.

        Raises:
            ValidationError:
                The mode is unknown, the reaction has neither reactants nor products, an entry is not a pair of a path
                and a state name, or a coefficient is not a positive number.
        """
        if self.mode not in MODES:
            error_msg = f'Reaction {self.label!r} has the mode {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)
        if not self.reactants and not self.products:
            error_msg = f'Reaction {self.label!r} has neither reactants nor products. Give at least one of them.'
            raise ValidationError(error_msg)
        for name in ('reactants', 'products', 'modifiers'):
            self._check_entries(name)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the reaction in its mode.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the reactants, the products, and
                the modifiers.

        Returns:
            laws:
                The rate laws, the jump law, or the rate and noise laws of the reaction.
        """
        if self.mode == 'jump':
            return super()._get_laws(namespace)
        propensity = self._get_rate(namespace)
        changes = self._get_changes(namespace)
        if self.mode == 'langevin':
            return self._get_langevin_laws(propensity, changes)
        return [RateLaw(state, change * propensity) for state, change in changes.items()]
