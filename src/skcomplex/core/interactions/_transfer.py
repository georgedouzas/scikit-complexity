"""Define transfers, which move amounts of states between members, the cells agents stand on, sources, and sinks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw, Sink, Source, TransferLaw
from ..laws.utils import where
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')
PAIRINGS = ('members', 'random', 'pairs', 'neighbors', 'cell', 'site')
SCHEDULES = ('random-sequential', 'member')
ACTORS = ('source', 'target')
REMOVALS = ('source', 'target')
LEG_LENGTH = 3

type Amount = float | Callable[..., object]
type Leg = tuple[str | Source, str | Sink | None, Amount]


class TransferInteraction(Interaction):
    """Transfer of amounts of states from the members of a source role to the members of a target role.

    A leg moves an amount out of a state of the source member, or out of `Source()`, into a state of the target member,
    or into `Sink()`. The transfer has a first leg, `state` into `into` by `amount`, and any further `legs`, each
    `(state, into, amount)`. The amounts of all legs are read before any of them moves, so a leg sees the states as
    they were before the transfer. A guard skips the transfer where it does not hold.

    The form chooses how the transfer lowers:

    - `'symbolic'` lowers to `TransferLaw`s between the source and the target of one element, or member `i` of both
      groups. The amounts, the guard, and the rate are numbers or functions of the namespace, and the roles are read as
      `namespace.source` and `namespace.target`. With a `rate`, the amount is the size of a jump at the rate.
    - `'agent'` lowers to an agent rule, whose amounts and guard are numbers or functions of the source agent, the
      target agent, and the model view. The pairing `'members'` pairs member `i` of both groups, `'random'` pairs every
      source member with a random target member, `'pairs'` draws disjoint random pairs, `'neighbors'` pairs every
      source member with the target members in the cells or nodes around it, and `'cell'` pairs the source and target
      members that stand in one cell or node. With the pairing `'site'` the target role binds a field, and every
      source member moves its amounts into the site of the field at its cell or node, so a negative amount takes from
      the site. The functions of this pairing take the cell of the source agent in place of a target agent.
      With `remove`, the source or the target member of every pair that transfers is removed, as an agent eaten whole.

    Under the schedule `'random-sequential'` the pairs act in random order. Under `'member'` every member of the
    `actor` role acts in its own activation, together with the rules of its element and the other member
    interactions, in the order the system lists them: it transfers with its partners, the members of the other role
    that the pairing `'random'`, `'neighbors'`, or `'cell'` gives it, or with `partners` of them drawn at random, and
    with itself under the pairing `'members'` within one group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the element or group the amounts leave.

        target:
            The path of the element or group the amounts enter, the source again for a transfer within members.

        state:
            The name of the state of the source the first leg takes from, or `Source()` for the environment.

        into:
            The name of the state of the target the first leg adds to, `Sink()` for the environment, or `None` for
            `state`.

        amount:
            The amount of the first leg, a number or a function.

        pairing:
            How members are paired, one of `'members'`, `'random'`, `'pairs'`, `'neighbors'`, or `'cell'`, the agent
            form alone taking the others than `'members'`.

        legs:
            Further legs, each a tuple of a state or `Source()`, a state, `Sink()`, or `None`, and an amount.

        guard:
            A function that returns whether a pair transfers, a condition of the namespace or a boolean of the agents,
            or `None` for every pair.

        rate:
            The rate of jumps of the symbolic form, a number or a function of the namespace, or `None` for a
            continuous flow.

        form:
            The lowering, `'symbolic'` to transfer laws or `'agent'` to an agent rule.

        schedule:
            The order of the agent form, `'random-sequential'` over the pairs, or `'member'` over the actors.

        actor:
            The role whose members act under the schedule `'member'`, `'source'` or `'target'`.

        partners:
            The number of partners an actor draws at random under the schedule `'member'`, or `None` for all of them.

        remove:
            The member of a pair that transfers that is removed, `'source'`, `'target'`, or `None`.

        phase:
            The phase the rule of the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The form, the pairing, the schedule, the actor, or the removal is unknown, a leg is malformed, the symbolic
            form has an agent pairing, the agent form a rate, or the schedule `'member'` a pairing it cannot draw,
            raised by `check`.

    Examples:
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.interactions import TransferInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> from typing import Annotated
        >>> class TankElement(Element):
        ...     def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.level = level
        >>> flow = TransferInteraction(
        ...     'flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2
        ... )
        >>> system = System(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
        >>> trajectory = Simulation(system).check().prepare().run(until=1.0).trajectory_
        >>> round(float(trajectory.values('upper.level')[-1] + trajectory.values('lower.level')[-1]), 6)
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        state: str | Source,
        into: str | Sink | None = None,
        amount: Amount,
        pairing: str = 'members',
        legs: Sequence[Leg] = (),
        guard: Callable[..., object] | None = None,
        rate: Amount | None = None,
        form: str = 'symbolic',
        schedule: str = 'random-sequential',
        actor: str = 'source',
        partners: int | None = None,
        remove: str | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.state = state
        self.into = into
        self.amount = amount
        self.pairing = pairing
        self.legs = legs
        self.guard = guard
        self.rate = rate
        self.form = form
        self.schedule = schedule
        self.actor = actor
        self.partners = partners
        self.remove = remove
        self.phase = phase

    def _get_legs(self) -> list[Leg]:
        """Return every leg with the state it adds to resolved."""
        legs: list[Leg] = []
        for state, into, amount in ((self.state, self.into, self.amount), *self.legs):
            legs.append((state, state if into is None and isinstance(state, str) else into, amount))
        return legs

    def _check_leg(self, state: object, into: object, amount: object) -> None:
        """Raise an error for a leg whose ends are not state names or the environment, or whose amount is not usable."""
        if not isinstance(state, str | Source) or not isinstance(into, str | Sink):
            error_msg = (
                f'A leg of {self.label!r} moves {state!r} into {into!r}. Give a state name or `Source()` it leaves, '
                'and a state name or `Sink()` it enters.'
            )
            raise ValidationError(error_msg)
        if isinstance(state, Source) and isinstance(into, Sink):
            error_msg = f'A leg of {self.label!r} moves from `Source()` into `Sink()`. Name a state of one end.'
            raise ValidationError(error_msg)
        if not callable(amount) and not isinstance(amount, int | float):
            error_msg = f'The amount {amount!r} of {self.label!r} is neither a number nor a function.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown form or pairing, a malformed leg, or an option the form does not take.

        Raises:
            ValidationError:
                The form or the pairing is unknown, a leg is malformed, the symbolic form has an agent pairing, or
                the agent form has a rate.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.pairing != 'members':
            error_msg = f'The symbolic form of {self.label!r} pairs members. Use `form="agent"` for {self.pairing!r}.'
            raise ValidationError(error_msg)
        if self.pairing not in PAIRINGS:
            error_msg = f'Unknown pairing {self.pairing!r} of {self.label!r}. Use one of {", ".join(PAIRINGS)}.'
            raise ValidationError(error_msg)
        for leg in self.legs:
            if not isinstance(leg, tuple) or len(leg) != LEG_LENGTH:
                error_msg = f'A leg of {self.label!r} is {leg!r}. Give a tuple `(state, into, amount)`.'
                raise ValidationError(error_msg)
        for state, into, amount in self._get_legs():
            self._check_leg(state, into, amount)
        if self.rate is not None and self.form == 'agent':
            error_msg = f'{self.label!r} has a rate, which only the symbolic form takes. Give `rate=None`.'
            raise ValidationError(error_msg)
        if self.guard is not None and not callable(self.guard):
            error_msg = f'The guard of {self.label!r} is {self.guard!r}. Give a function, or None.'
            raise ValidationError(error_msg)
        self._check_schedule()

    def _check_schedule(self) -> None:
        """Raise an error for an unknown schedule, actor, or removal, or partners the schedule cannot draw."""
        for name, value, variants in (
            ('schedule', self.schedule, SCHEDULES),
            ('actor', self.actor, ACTORS),
            ('removal', self.remove, (*REMOVALS, None)),
        ):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {", ".join(map(str, variants))}.'
                raise ValidationError(error_msg)
        own = self.pairing == 'members' and self.source == self.target
        if self.schedule == 'member' and self.pairing in {'members', 'pairs'} and not own:
            error_msg = (
                f'The actors of {self.label!r} draw partners by `random`, `neighbors`, `cell`, or `site`, or pair '
                'with themselves by `members` within one group.'
            )
            raise ValidationError(error_msg)
        if self.pairing == 'site' and self.actor != 'source':
            error_msg = f'The sources of {self.label!r} move amounts into a field. Give `actor="source"`.'
            raise ValidationError(error_msg)
        if self.schedule == 'member' and self.form != 'agent':
            error_msg = f'The schedule `member` of {self.label!r} runs agent rules. Use `form="agent"`.'
            raise ValidationError(error_msg)
        if self.partners is not None and (isinstance(self.partners, bool) or not self.partners >= 1):
            error_msg = f'The partners of {self.label!r} are {self.partners!r}. Give a positive whole number, or None.'
            raise ValidationError(error_msg)

    def _get_law(self, namespace: SimpleNamespace, leg: Leg) -> TransferLaw:
        """Return the transfer law of one leg between the members of a namespace."""
        state, into, amount = leg
        held = state if isinstance(state, Source) else getattr(namespace.source, state)
        sink = into if isinstance(into, Sink) else getattr(namespace.target, str(into))
        value = amount(namespace) if callable(amount) else amount
        if self.rate is None:
            if self.guard is not None:
                value = where(self.guard(namespace), value, 0)
            return TransferLaw(held, sink, value)
        rate = self.rate(namespace) if callable(self.rate) else self.rate
        if self.guard is not None:
            rate = where(self.guard(namespace), rate, 0)
        return TransferLaw(held, sink, value, rate)

    def _get_pairs(self, model: ModelViewProtocol) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of source and target agents of one step in random order."""
        sources = model.agents('source')
        targets = sources if self.source == self.target else model.agents('target')
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        if self.pairing == 'members':
            pairs = list(zip(sources, targets, strict=False))
        elif self.pairing == 'random':
            pairs = self._get_random_pairs(sources, targets, model)
        elif self.pairing == 'pairs':
            pairs = self._get_disjoint_pairs(sources, targets, model)
        else:
            pairs = self._get_located_pairs(sources, targets)
        return [pairs[int(index)] for index in model.rng.permutation(len(pairs))]

    @staticmethod
    def _get_random_pairs(
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return every source agent paired with a random target agent other than itself."""
        pairs = []
        for source in sources:
            others = [target for target in targets if target is not source]
            if others:
                pairs.append((source, others[int(model.rng.integers(len(others)))]))
        return pairs

    def _get_disjoint_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return disjoint random pairs, of two halves of one group or of a source and a target agent."""
        first = [sources[int(index)] for index in model.rng.permutation(len(sources))]
        if self.source == self.target:
            return list(zip(first[::2], first[1::2], strict=False))
        second = [targets[int(index)] for index in model.rng.permutation(len(targets))]
        return list(zip(first, second, strict=False))

    def _get_located_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of agents in one cell, or of every source agent with the target agents around it."""
        located: dict[object, list[AgentViewProtocol]] = {}
        for target in targets:
            located.setdefault(target.location_key, []).append(target)
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        for source in sources:
            keys = [source.location_key] if self.pairing == 'cell' else source.neighbor_cells()
            pairs.extend((source, target) for key in keys for target in located.get(key, []) if target is not source)
        return pairs

    def _evaluate(
        self,
        amount: Amount,
        source: AgentViewProtocol,
        target: object,
        model: ModelViewProtocol,
    ) -> float:
        """Return the amount of a leg between two agents as a number."""
        value = amount(source, target, model) if callable(amount) else amount
        if not isinstance(value, int | float | np.floating | np.integer):
            error_msg = f'An amount of {self.label!r} between agents is {value!r}. Return a number.'
            raise ValidationError(error_msg)
        return float(value)

    def _transfer_pair(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Transfer between two living agents, and remove the member of the pair that a removal names."""
        if source.alive and target.alive and self.transfer(source, target, model) and self.remove is not None:
            (source if self.remove == 'source' else target).remove()

    def _deposit(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Move the amounts of every leg from an agent into the site of the target field at its cell or node."""
        cell = agent.location_key
        if not agent.alive or (self.guard is not None and not self.guard(agent, cell, model)):
            return
        legs = self._get_legs()
        amounts = [self._evaluate(amount, agent, cell, model) for _, _, amount in legs]
        for (state, into, _), amount in zip(legs, amounts, strict=True):
            if isinstance(state, str):
                setattr(agent, state, getattr(agent, state) - amount)
            if isinstance(into, str):
                model.deposit(f'target.{into}', cell, amount)

    def _get_partners(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> list[AgentViewProtocol]:
        """Return the partners of an actor by the pairing, or the number of them drawn at random."""
        if self.pairing == 'members':
            return [agent]
        role = 'source' if self.actor == 'target' else 'target'
        if self.pairing == 'cell':
            groups = set(model.groups(role))
            others = [other for other in agent.agents_here() if other.group in groups]
        else:
            others = [other for other in model.agents(role) if other is not agent]
        if self.pairing == 'neighbors':
            cells = set(agent.neighbor_cells())
            others = [other for other in others if other.location_key in cells]
        elif self.pairing == 'random' and others:
            others = [others[int(model.rng.integers(len(others)))]]
        if self.partners is None or len(others) <= self.partners:
            return others
        return [others[int(index)] for index in model.rng.permutation(len(others))[: self.partners]]

    def _transfer_member(self, agent: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Transfer between one actor and its partners, in the activation of the actor."""
        if self.pairing == 'site':
            self._deposit(agent, model)
            return
        for partner in self._get_partners(agent, model):
            if agent.alive:
                pair = (partner, agent) if self.actor == 'target' else (agent, partner)
                self._transfer_pair(*pair, model)

    def _transfer_pairs(self, model: ModelViewProtocol) -> None:
        """Transfer between every pair of agents of one step, or into the sites of the source agents."""
        if self.pairing == 'site':
            sources = model.agents('source')
            for index in model.rng.permutation(len(sources)):
                self._deposit(sources[int(index)], model)
            return
        for source, target in self._get_pairs(model):
            self._transfer_pair(source, target, model)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer laws of the symbolic form, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                A transfer law per leg, or the rule that transfers between the pairs of agents.
        """
        if self.form == 'agent' and self.schedule == 'member':
            return [RuleLaw(self._transfer_member, phase=self.phase, members=self.actor)]
        if self.form == 'agent':
            return [RuleLaw(self._transfer_pairs, phase=self.phase)]
        return [self._get_law(namespace, leg) for leg in self._get_legs()]

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> bool:
        """Move the amounts of every leg from one source agent to one target agent, when the guard holds.

        Args:
            source:
                The agent the amounts leave.

            target:
                The agent the amounts enter.

            model:
                The model view of the rule.

        Returns:
            transferred:
                Whether the guard held, so the amounts moved.

        Raises:
            ValidationError:
                An amount function returns something other than a number.
        """
        if self.guard is not None and not self.guard(source, target, model):
            return False
        legs = self._get_legs()
        amounts = [self._evaluate(amount, source, target, model) for _, _, amount in legs]
        for (state, into, _), amount in zip(legs, amounts, strict=True):
            if isinstance(state, str):
                setattr(source, state, getattr(source, state) - amount)
            if isinstance(into, str):
                setattr(target, into, getattr(target, into) + amount)
        return True
