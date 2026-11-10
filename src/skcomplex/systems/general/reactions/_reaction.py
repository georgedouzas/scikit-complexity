"""Define reactions.

Reactions are jumps that move units between states by a product propensity or a given propensity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar, SupportsFloat

from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.base import BasePart
from ....core.components.interactions import SCHEDULES, JumpInteraction
from ....core.components.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, RuleLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, Tags
from ....core.types import ElementKind
from .types import Stoichiometry

MODES = ('rate', 'jump', 'langevin')
FORMS = ('symbolic', 'agent')
AGENT_MODES = ('rate', 'jump')
ENTRY_SIZE = 2


class ReactionInteraction(JumpInteraction):
    """Reaction that moves units from its reactants to its products.

    It moves them at a product propensity or a given propensity. A reactant, a product, or a modifier is a pair of the
    path of an element, or of a member of a group such as `cells[0]`, and the name of its state that holds the units,
    mapped to its stoichiometric coefficient. A coefficient is a positive number, or a function of the namespace of the
    reaction that returns an expression of its parameters, such as `lambda s: 1 / s.target.epsilon`. So `S + I -> 2 I`
    has the reactants `{('S', 'n'): 1, ('I', 'n'): 1}` and the products `{('I', 'n'): 2}`, and two states of one element
    are `('cell', 'u')` and `('cell', 'v')`. A species may also be the name of a state of the `target` alone, such as
    `{'u': 1, 'v': 2}`. A target that is a group or a field reacts in every member, with the species of every member its
    own states, as the reactions at every site of a reaction-diffusion field.

    A subclass may bind its species under roles of its own names, such as `susceptible` and `infected`: it returns its
    entries from `_get_reactants`, `_get_products`, and `_get_modifiers`, pairs of the path a role binds and a state
    name, and the reaction reads every pair from the namespace of the role that binds its path.

    The default propensity is the rate constant `k` times the product of the reactant and modifier amounts, each raised
    to its coefficient, or else the expression the `propensity` function returns. Fields name this product propensity in
    their own words, such as the law of mass action in chemistry, homogeneous mixing in epidemiology, and the encounter
    term of Lotka-Volterra in ecology. The reaction changes every state by its coefficient among the products minus its
    coefficient among the reactants, so it conserves the total of the states whose changes sum to zero, and modifiers
    enter the propensity but do not change.

    In the mode `'rate'` the reaction is a rate equation. In the mode `'jump'` it is the random event of a
    `JumpInteraction` at the propensity that moves the units of the coefficients. In the mode `'langevin'` it is the
    chemical Langevin equation, the rate equation with one Wiener process of amplitude the square root of the propensity
    driving every state the reaction changes.

    In the `'agent'` form the target is an agent group and every member reacts on its own states, the species named by
    state names alone, in every step from the states at its start. The propensity of a member is `k` times the product
    of its reactant and modifier states to the power of their coefficients, or the number `propensity(agent, system)`
    returns, and a coefficient that is a function is the number `coefficient(agent, system)` returns. In the mode
    `'rate'` the member adds the changes times its propensity, one unit step of the rate equation, and in the mode
    `'jump'` it adds the changes with the probability `1 - exp(-propensity)` that an event fires in the step. The
    members react at once from the states at the start of the rule under the schedule `'synchronous'`, one after the
    other in their order or in a new random order every step under `'sequential'` or `'random-sequential'`, or each in
    its own activation under `'member'`.

    Args:
        label:
            The label of the reaction, unique among the parts of its system.

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
            its members, and the symbol `k`. In the agent form, a function `propensity(agent, system)` that returns a
            number.

        modifiers:
            The coefficients of the modifiers, whose states the propensity reads but the reaction does not change, or
            `None`.

        mode:
            The form of the laws, `'rate'`, `'jump'`, or `'langevin'`.

        form:
            `'symbolic'` for the laws of the states, or `'agent'` for a rule in which every member of an agent group
            reacts on its own states.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

        schedule:
            The order of the members in the agent form, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`.

    Raises:
        ValidationError:
            The mode, the form, or the schedule is unknown, the symbolic form is not synchronous, the reaction has
            neither reactants nor products, an entry is neither a pair of a path and a state name nor a state name of a
            given target, a coefficient is neither a positive number nor a function, or the agent form has no target,
            has an entry that is a pair, or has the mode `'langevin'`.

    Examples:
        >>> from skcomplex.systems.general.elements import NonnegativeElement
        >>> from skcomplex.systems.general.reactions import ReactionInteraction
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> infection = ReactionInteraction(
        ...     'infection', reactants={('S', 'x'): 1, ('I', 'x'): 1}, products={('I', 'x'): 2}, k=0.0005
        ... )
        >>> recovery = ReactionInteraction('recovery', reactants={('I', 'x'): 1}, products={('R', 'x'): 1}, k=0.1)
        >>> system = CompositeSystem(
        ...     elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')],
        ...     interactions=[infection, recovery],
        ... )
        >>> states = simulate(system, until=50, dt=1.0).states
        >>> round(float(states[-1].sum()), 6)
        1000.0

        The decay `x -> 0` at every site of a field, at the rate `k x` of the site:

        >>> import numpy as np
        >>> from skcomplex.core.components.elements import FieldElement
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> decay = ReactionInteraction('decay', target='cells', reactants={'x': 1}, k=1.0)
        >>> values = {'x': np.arange(3.0)}
        >>> cells = FieldElement('cells', element=NonnegativeElement, space=LatticeSpace((3,)), values=values)
        >>> trajectory = simulate(CompositeSystem(elements=[cells], interactions=[decay]), until=1, dt=1.0)
        >>> np.round(trajectory.lattice('cells.x')[-1], 3).tolist()
        [0.0, 0.368, 0.736]

        A coefficient that is a parameter, the decay at the rate `x` that removes `1 / epsilon` units per event:

        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> class ScaledElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         x: Annotated[float, StateSpec()] = 1.0,
        ...         epsilon: Annotated[float, ParameterSpec()] = 0.5,
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.epsilon = epsilon
        >>> scaled = ReactionInteraction(
        ...     'decay', target='box', reactants={'x': lambda s: 1 / s.target.epsilon}, propensity=lambda s: s.target.x
        ... )
        >>> system = CompositeSystem(elements=[ScaledElement('box')], interactions=[scaled])
        >>> trajectory = simulate(system, until=0.5, dt=0.01)
        >>> round(float(trajectory.get_values('box.x')[-1]), 3)
        0.368

        Every agent of a group decays on its own states, by `k n` in a step of the rate equation:

        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class CellElement(Element):
        ...     def __init__(self, label, n: Annotated[float, StateSpec()] = 3.0):
        ...         super().__init__(label)
        ...         self.n = n
        >>> cells = GroupElement('cells', element=CellElement, size=2)
        >>> decay = ReactionInteraction('decay', target='cells', reactants={'n': 1}, k=0.1, form='agent')
        >>> system = CompositeSystem(elements=[cells], interactions=[decay], time=DiscreteTime())
        >>> trajectory = simulate(system, until=2)
        >>> np.round(trajectory.get_agents('cells', 2).values['n'], 6).tolist()
        [2.43, 2.43]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw, NoiseLaw, JumpLaw, RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the agent-form rules, or the laws of the symbolic one.

        The laws of the symbolic form are rate, jump, or Langevin laws.
        """
        if self._get_argument('form') == 'agent':
            return self._get_variant_tags((RuleLaw,), stochastic=self._get_mode() == 'jump')
        modes: dict[object, tuple[type[BaseLaw], ...]] = {'jump': (JumpLaw,), 'langevin': (RateLaw, NoiseLaw)}
        return self._get_variant_tags(modes.get(self._get_mode(), (RateLaw,)))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[
            str | None,
            RoleSpec(elements=(ElementKind.ANY,), states=('reactants', 'products', 'modifiers')),
        ] = None,
        reactants: Annotated[Stoichiometry | None, RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = None,
        products: Annotated[Stoichiometry | None, RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        propensity: Callable[..., object] | None = None,
        modifiers: Annotated[Stoichiometry | None, RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = None,
        mode: str = 'rate',
        form: str = 'symbolic',
        phase: str | None = None,
        schedule: str = 'synchronous',
    ) -> None:
        super().__init__(label, target='')
        self.target = target
        self.reactants = reactants
        self.products = products
        self.k = k
        self.propensity = propensity
        self.modifiers = modifiers
        self.mode = mode
        self.form = form
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
        roles = [
            name
            for name, spec in self._get_specs().items()
            if isinstance(spec, RoleSpec) and not spec.variadic and not spec.noise
        ]
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

    def _get_langevin_laws(self, propensity: Expression, changes: dict[Expression, float]) -> list[BasePart]:
        """Return the drift of the reaction and the noise of its Wiener process.

        The noise of its one Wiener process acts on every state it changes.
        """
        amplitude = sqrt(where(propensity > 0, propensity, 0))
        rates: list[BasePart] = [RateLaw(state, change * propensity) for state, change in changes.items()]
        noises: list[BasePart] = [
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

    def _check_form(self) -> None:
        """Raise an error for an unknown form, or a bad agent form.

        A bad agent form has no target, has pairs, or is of Langevin mode.
        """
        if self.form not in FORMS:
            error_msg = f'Reaction {self.label!r} has the form {self.form!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        self._check_variant('schedule', self.schedule, SCHEDULES)
        if self.form != 'agent' and self.schedule != 'synchronous':
            error_msg = f'The schedule {self.schedule!r} of reaction {self.label!r} runs agent rules. Use form="agent".'
            raise ValidationError(error_msg)
        if self.form != 'agent':
            return
        if self.target is None:
            error_msg = f'Reaction {self.label!r} has the agent form but no target. Give the agent group as `target`.'
            raise ValidationError(error_msg)
        mode = self._get_mode()
        if mode not in AGENT_MODES:
            error_msg = (
                f'Reaction {self.label!r} has the agent form and the mode {mode!r}. Use {" or ".join(AGENT_MODES)}.'
            )
            raise ValidationError(error_msg)
        entries = [*(self._get_reactants() or {}), *(self._get_products() or {}), *(self._get_modifiers() or {})]
        if not all(isinstance(entry, str) for entry in entries):
            error_msg = (
                f'Reaction {self.label!r} has the agent form, whose species are states of every member. Name them by '
                'state names alone, such as {"n": 1}.'
            )
            raise ValidationError(error_msg)

    def _check_jump(self) -> None:
        """Raise an error for an unknown mode or form, no entries, or bad entries.

        Raises:
            ValidationError:
                The mode or the form is unknown, the reaction has neither reactants nor products, an entry is neither a
                pair of a path and a state name nor a state name of a given target, a coefficient is neither a positive
                number nor a function, or the agent form has no target, has an entry that is a pair, or has the mode
                `'langevin'`.
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
        self._check_form()

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

        Args:
            system:
                The system view of the interaction.
        """
        self._update_agents(system, self.schedule, self._get_agent_step, self._apply_agent_step)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the reaction in its mode, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespaces of the target, the reactants, the
                products, and the modifiers.

        Returns:
            laws:
                The rate laws, the jump law, the rate and noise laws, or the rule of the reaction.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._react_member, phase=self.phase, members=self._get_role('target'))]
        if self.form == 'agent':
            return [RuleLaw(self.react, phase=self.phase)]
        mode = self._get_mode()
        if mode == 'jump':
            return [*super()._get_laws(namespace)]
        propensity = self._get_rate(namespace)
        changes = self._get_changes(namespace)
        if mode == 'langevin':
            return self._get_langevin_laws(propensity, changes)
        return [RateLaw(state, change * propensity) for state, change in changes.items()]
