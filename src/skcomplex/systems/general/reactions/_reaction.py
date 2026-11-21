"""Define the interaction that moves units from reactants to products."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar, SupportsFloat

from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Tags
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.enums import ScheduleKind
from ....core.errors import ValidationError
from ....core.types import Role
from .types import Stoichiometry

MODES = ('rate', 'jump', 'langevin')
AGENT_MODES = ('rate', 'jump')
ENTRY_SIZE = 2


class ReactionInteraction(Interaction):
    """An interaction that moves units from its reactants to its products.

    A reactant, a product, or a modifier is a pair of the path of an element and the name of its state, mapped to its
    stoichiometric coefficient, the number of units the reaction takes or gives. So `S + I -> 2 I` has the reactants
    `{('S', 'n'): 1, ('I', 'n'): 1}` and the products `{('I', 'n'): 2}`. A coefficient is a positive number or a
    function of the namespace. A species may also be the name of a state of `target` alone, such as `{'u': 1, 'v': 2}`,
    and a target that is a group or a field reacts in every member. A subclass may bind its species under roles of its
    own, such as `susceptible` and `infected`, and return its entries from `_get_reactants`, `_get_products`, and
    `_get_modifiers`.

    The propensity is the rate constant `k` times the product of the reactant and modifier amounts, each raised to its
    coefficient, or the expression that `propensity` returns. Fields name this product their own way: mass action in
    chemistry, homogeneous mixing in epidemiology, the encounter term of Lotka and Volterra in ecology. The `mode` says
    how the reaction runs:

    - `'rate'`: a rate equation.
    - `'jump'`: a random event at the propensity that moves the units of the coefficients.
    - `'langevin'`: the chemical Langevin equation, the rate equation with a noise of amplitude the square root of the
      propensity.

    With a `schedule` it sets one rule that walks the members of an agent group in that order, and every member reacts
    on its own states at every step.

    Unlike `Interaction`, which takes laws written by hand, it builds the changes from a stoichiometric vector and the
    propensity from mass action, and the same reaction runs as a rate, a jump, or a Langevin equation. Choose it when a
    model is written as reactions between species.

    It appears in many fields. Molecules react in a cell, people pass from susceptible to infected in an epidemic,
    predators eat prey, and enzymes bind their substrates.

    Args:
        label:
            The label of the reaction, unique among the components of its system.

        target:
            The path of the element, group, or field whose states the species named by a state alone are, or `None`.

        reactants:
            The coefficients of the reactants, or `None` for a source.

        products:
            The coefficients of the products, or `None` for a sink.

        k:
            The rate constant.

        propensity:
            A function of the namespace of the reaction that returns the propensity, or `None` for the product
            propensity. The namespace holds the tuples `reactants`, `products`, and `modifiers` of the namespaces of the
            elements their pairs bind, in the order of their entries, `target`, the namespace of the target or of one of
            its members, and the symbol `k`. With a schedule, a function `propensity(agent, system)` that returns a
            number.

        modifiers:
            The coefficients of the modifiers, whose states the propensity reads but the reaction does not change, or
            `None`.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

        phase:
            The phase of the step the rule of a schedule runs in, or `None` for the default phase.

        schedule:
            The order in which a rule walks the members, `'synchronous'` for every member at once, `'sequential'` or
            `'random-sequential'` for one member after the other, or `'member'` for every member in its own activation,
            or `None` for the laws of the states.

    Raises:
        ValidationError:
            The mode or the schedule is unknown, the reaction has neither reactants nor products, an entry is neither a
            pair of a path and a state name nor a state name of a given target, a coefficient is neither a positive
            number nor a function, or a reaction with a schedule has no target, has an entry that is a pair, or has
            the mode `'langevin'`.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.elements import NonnegativeElement
        >>> from skcomplex.systems.general.reactions import ReactionInteraction
        >>> infection = ReactionInteraction(
        ...     'infection', reactants={('S', 'x'): 1, ('I', 'x'): 1}, products={('I', 'x'): 2}, k=0.0005
        ... )
        >>> recovery = ReactionInteraction('recovery', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=0.1)
        >>> system = System(
        ...     elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')],
        ...     interactions=[infection, recovery],
        ... )
        >>> states = system.prepare().simulate(until=50, dt=1.0).states
        >>> round(float(states[-1].sum()), 6)
        1000.0

        The decay `x -> 0` at every site of a lattice, at the rate `k x` of the site:

        >>> import numpy as np
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> decay = ReactionInteraction('decay', target='cells', reactants={'x': 1}, k=1.0)
        >>> cells = NonnegativeElement('cells', x=np.arange(3.0), space=LatticeSpace((3,)))
        >>> trajectory = System(elements=[cells], interactions=[decay]).prepare().simulate(until=1, dt=1.0)
        >>> np.round(trajectory.get_lattice('cells.x')[-1], 3).tolist()
        [0.0, 0.368, 0.736]

        A coefficient that is a parameter, the decay at the rate `x` that removes `1 / epsilon` units per event:

        >>> from skcomplex.core.components.elements import Element
        >>> scaled = ReactionInteraction(
        ...     'decay', target='box', reactants={'x': lambda s: 1 / s.target.epsilon}, propensity=lambda s: s.target.x
        ... )
        >>> box = Element('box', values={'x': 1.0, 'epsilon': 0.5})
        >>> trajectory = System(elements=[box], interactions=[scaled]).prepare().simulate(until=0.5, dt=0.01)
        >>> round(float(trajectory.get_values('box.x')[-1]), 3)
        0.368

        Every agent of a group decays on its own states, by `k n` in a step of the rate equation:

        >>> cells = Element('cells', values={'n': 3.0}, size=2)
        >>> decay = ReactionInteraction('decay', target='cells', reactants={'n': 1}, k=0.1, schedule='synchronous')
        >>> trajectory = System(elements=[cells], interactions=[decay]).prepare().simulate(until=2)
        >>> np.round(trajectory.get_agents('cells', 2).values['n'], 6).tolist()
        [2.43, 2.43]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, NoiseLaw, JumpLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rule of a schedule, or of the laws without one.

        The laws without a schedule are rate, jump, or Langevin laws.
        """
        if self._get_argument('schedule') is not None:
            return self._get_variant_tags((UpdateLaw,), stochastic=self._get_mode() == 'jump')
        modes: dict[object, tuple[type[BaseLaw], ...]] = {'jump': (JumpLaw,), 'langevin': (RateLaw, NoiseLaw)}
        return self._get_variant_tags(modes.get(self._get_mode(), (RateLaw,)))

    def __init__(
        self,
        label: str,
        *,
        target: Role | None = None,
        reactants: Stoichiometry | None = None,
        products: Stoichiometry | None = None,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        propensity: Callable[..., object] | None = None,
        modifiers: Stoichiometry | None = None,
        mode: str = 'rate',
        phase: str | None = None,
        schedule: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.reactants = reactants
        self.products = products
        self.k = k
        self.propensity = propensity
        self.modifiers = modifiers
        self.mode = mode
        self.phase = phase
        self.schedule = schedule

    def _get_reactants(self) -> Stoichiometry | None:
        """Return the coefficients of the reactants.

        A subclass with roles of its own names returns them.
        """
        return self.reactants

    def _get_products(self) -> Stoichiometry | None:
        """Return the coefficients of the products.

        A subclass with roles of its own names returns them.
        """
        return self.products

    def _get_modifiers(self) -> Stoichiometry | None:
        """Return the coefficients of the modifiers.

        A subclass with roles of its own names returns them.
        """
        return self.modifiers

    def _get_mode(self) -> str:
        """Return the form of the laws.

        A subclass that sets it by another argument returns it.
        """
        return self.mode

    def _get_bound(self, namespace: SimpleNamespace) -> dict[str, SimpleNamespace]:
        """Return the namespace of every path the plain roles of the reaction bind."""
        roles = self._get_roles(single=True)
        return {
            getattr(self, name): getattr(namespace, name)
            for name in roles
            if isinstance(getattr(self, name), str) and hasattr(namespace, name)
        }

    def _get_terms(
        self,
        entries: Stoichiometry | None,
        namespace: SimpleNamespace,
        name: str,
    ) -> list[tuple[Expression, Expression | float]]:
        """Return the state symbol and the coefficient of every entry of a kind of species.

        A state name alone is a state of the target. A pair reads the namespace of the plain role that binds its path,
        or else the namespace of its variadic role `name`, in the order of the pairs. A coefficient that is a function
        is its expression at the namespace.
        """
        entries = entries or {}
        pairs = [entry for entry in entries if not isinstance(entry, str)]
        sources: dict[object, SimpleNamespace] = {}
        if pairs and name in self._get_specs():
            sources = dict(zip(pairs, getattr(namespace, name), strict=True))
        elif pairs:
            bound = self._get_bound(namespace)
            sources = {pair: bound[pair[0]] for pair in pairs}
        return [
            (
                getattr(
                    self._get_member(namespace, 'target') if isinstance(entry, str) else sources[entry],
                    self._get_state(entry),
                ),
                SR(coefficient(namespace)) if callable(coefficient) else coefficient,
            )
            for entry, coefficient in entries.items()
        ]

    def _get_state(self, entry: tuple[str, str] | str) -> str:
        """Return the name of the state of an entry.

        The entry is a state name alone or the second item of a pair.
        """
        return entry if isinstance(entry, str) else entry[1]

    def _get_changes(self, namespace: SimpleNamespace) -> dict[Expression, float]:
        """Return the net change of every state the reaction changes.

        It is the products minus the reactants.
        """
        changes: dict[Expression, float] = {}
        for sign, entries, name in (
            (-1, self._get_reactants(), 'reactants'),
            (1, self._get_products(), 'products'),
        ):
            for state, coefficient in self._get_terms(entries, namespace, name):
                changes[state] = changes.get(state, 0) + sign * coefficient
        return {state: change for state, change in changes.items() if change}

    def _get_product_propensity(self, namespace: SimpleNamespace, constant: Expression) -> Expression:
        """Return the product propensity of a rate constant.

        It is the rate constant times the product of the reactant and modifier amounts, each raised to its coefficient.
        """
        propensity = SR(constant)
        for entries, name in ((self._get_reactants(), 'reactants'), (self._get_modifiers(), 'modifiers')):
            for state, coefficient in self._get_terms(entries, namespace, name):
                propensity *= state**coefficient
        return propensity

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the propensity of the reaction.

        It is given by the user's function or by the product propensity at the rate constant `k`.
        """
        if self.propensity is not None:
            return SR(self.propensity(namespace))
        return self._get_product_propensity(namespace, self._get_parameter_symbol('k', namespace))

    def _get_langevin_laws(self, propensity: Expression, changes: dict[Expression, float]) -> list[BaseLaw]:
        """Return the drift of the reaction and the noise of its Wiener process.

        The noise of its one Wiener process acts on every state it changes.
        """
        amplitude = sqrt(where(propensity > 0, propensity, 0))
        rates: list[BaseLaw] = [RateLaw(state, change * propensity) for state, change in changes.items()]
        noises: list[BaseLaw] = [
            NoiseLaw(state, change * amplitude, process=self.label) for state, change in changes.items()
        ]
        return rates + noises

    def _is_entry(self, entry: object) -> bool:
        """Check whether an entry is a pair of a path and a state name, or a state name.

        The state name is that of a given target.
        """
        if isinstance(entry, str):
            return self.target is not None and entry.isidentifier()
        return (
            isinstance(entry, tuple)
            and len(entry) == ENTRY_SIZE
            and all(isinstance(item, str) for item in entry)
            and entry[1].isidentifier()
        )

    def _is_coefficient(self, coefficient: object) -> bool:
        """Check whether a coefficient is a positive number or a function."""
        return callable(coefficient) or (isinstance(coefficient, int | float) and coefficient > 0)

    def _check_entries(self, name: str, entries: Stoichiometry | None) -> None:
        """Raise an error for entries that are not pairs or target states, or coefficients.

        The states are state names of the target.
        """
        if entries is None:
            return
        valid = isinstance(entries, Mapping) and all(
            self._is_entry(entry) and self._is_coefficient(coefficient) for entry, coefficient in entries.items()
        )
        if not valid:
            error_msg = (
                f'The {name} of reaction {self.label!r} are {entries!r}. Map pairs of a path and a state name, or '
                'state names of the target, to positive numbers or functions, such as {("S", "n"): 1} or {"n": 1}.'
            )
            raise ValidationError(error_msg)

    def _check_schedule(self) -> None:
        """Raise an error for an unknown schedule, or a bad reaction with one.

        A bad reaction with a schedule has no target, has pairs, or is of Langevin mode.
        """
        if self.schedule is None:
            return
        self._check_variant('schedule', self.schedule, ScheduleKind)
        if self.target is None:
            error_msg = (
                f'Reaction {self.label!r} has the schedule {self.schedule!r} but no target. Give the agent group as '
                '`target`.'
            )
            raise ValidationError(error_msg)
        mode = self._get_mode()
        if mode not in AGENT_MODES:
            error_msg = (
                f'Reaction {self.label!r} has the schedule {self.schedule!r} and the mode {mode!r}. Use '
                f'{" or ".join(AGENT_MODES)}.'
            )
            raise ValidationError(error_msg)
        entries = [*(self._get_reactants() or {}), *(self._get_products() or {}), *(self._get_modifiers() or {})]
        if not all(isinstance(entry, str) for entry in entries):
            error_msg = (
                f'Reaction {self.label!r} has a schedule, so its species are states of every member. Name them by '
                'state names alone, such as {"n": 1}.'
            )
            raise ValidationError(error_msg)

    def _check_jump(self) -> None:
        """Raise an error for an unknown mode or schedule, no entries, or bad entries.

        Raises:
            ValidationError:
                The mode or the schedule is unknown, the reaction has neither reactants nor products, an entry is
                neither a pair of a path and a state name nor a state name of a given target, a coefficient is neither a
                positive number nor a function, or a reaction with a schedule has no target, has an entry that is a
                pair, or has the mode `'langevin'`.
        """
        mode = self._get_mode()
        if mode not in MODES:
            error_msg = f'Reaction {self.label!r} has the mode {mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)
        reactants, products = self._get_reactants(), self._get_products()
        if not reactants and not products:
            error_msg = f'Reaction {self.label!r} has neither reactants nor products. Give at least one of them.'
            raise ValidationError(error_msg)
        for name, entries in (('reactants', reactants), ('products', products), ('modifiers', self._get_modifiers())):
            self._check_entries(name, entries)
        self._check_schedule()

    def _get_number(self, value: object, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return a number, or the value of a function of an agent and the system.

        The value is returned as a number.
        """
        found = value(agent, system) if callable(value) else value
        if not isinstance(found, SupportsFloat):
            error_msg = f'A function of reaction {self.label!r} returns {found!r} for an agent. Return a number.'
            raise ValidationError(error_msg)
        return float(found)

    def _get_agent_changes(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> dict[str, float]:
        """Return the net change of every state of an agent the reaction changes.

        It is the products minus the reactants.
        """
        changes: dict[str, float] = {}
        for sign, entries in ((-1, self._get_reactants()), (1, self._get_products())):
            for entry, coefficient in (entries or {}).items():
                state = self._get_state(entry)
                changes[state] = changes.get(state, 0.0) + sign * self._get_number(coefficient, agent, system)
        return changes

    def _get_agent_propensity(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the propensity of an agent.

        It is given by the user's function or by the product propensity at the rate constant `k`.
        """
        if self.propensity is not None:
            return self._get_number(self.propensity, agent, system)
        propensity = self._get_parameter_value('k', system)
        for entries in (self._get_reactants(), self._get_modifiers()):
            for entry, coefficient in (entries or {}).items():
                value = float(getattr(agent, self._get_state(entry)))
                propensity *= value ** self._get_number(coefficient, agent, system)
        return propensity

    def _get_agent_scale(self, propensity: float, system: SystemViewProtocol) -> float:
        """Return the multiple of its changes an agent adds in a step.

        It is its propensity or whether an event fires.
        """
        if self._get_mode() == 'rate':
            return propensity
        return float(system.rng.random() < 1.0 - math.exp(-max(propensity, 0.0)))

    def _get_agent_step(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> dict[str, float]:
        """Return the change of every state of an agent in the step.

        It is its changes times its propensity or its event.
        """
        changes, propensity = self._get_agent_changes(agent, system), self._get_agent_propensity(agent, system)
        scale = self._get_agent_scale(propensity, system)
        return {state: change * scale for state, change in changes.items()}

    def _apply_agent_step(self, agent: AgentViewProtocol, changes: dict[str, float]) -> None:
        """Add their changes to the states of an agent."""
        for state, change in changes.items():
            setattr(agent, state, float(getattr(agent, state)) + change)

    def _react_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """React in one agent on its own states, in its own activation."""
        if agent.alive:
            self._apply_agent_step(agent, self._get_agent_step(agent, system))

    def react(self, system: SystemViewProtocol) -> None:
        """React in every agent of the target on its own states, by the schedule.

        Without a schedule every agent reacts at once.

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule or 'synchronous', self._get_agent_step, self._apply_agent_step)

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The mode or the schedule is unknown, the reaction has neither reactants nor products, or an entry or a
                coefficient is not valid.
        """
        super()._check_arguments()
        self._check_jump()

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set its laws or its agent rule for a binding.

        Args:
            context:
                The namespace of the target, the reactants, the products, and the modifiers.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = self._get_reaction_laws(context)

    def _get_reaction_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the reaction in its mode, or the rule of its schedule.

        The rule of a schedule changes one member, or every member of the target in the order of the schedule.
        """
        if self.schedule == 'member':
            return [UpdateLaw(self._get_member(namespace, 'target'), self._react_member, phase=self.phase)]
        if self.schedule is not None:
            return [UpdateLaw(namespace, self.react, phase=self.phase)]
        propensity = self._get_rate(namespace)
        changes = self._get_changes(namespace)
        mode = self._get_mode()
        if mode == 'jump':
            return [JumpLaw(propensity, changes)]
        if mode == 'langevin':
            return self._get_langevin_laws(propensity, changes)
        return [RateLaw(state, change * propensity) for state, change in changes.items()]
