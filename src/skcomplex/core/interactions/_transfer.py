"""Define transfers, which move amounts of states between members, the cells agents stand on, sources, and sinks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ...errors import SimulationError, ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw, Sink, Source, TransferLaw
from ..laws.utils import where
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')
PAIRINGS = ('members', 'random', 'pairs', 'neighbors', 'neighbor-pairs', 'cell')
LEG_LENGTHS = (3, 4)

type Amount = float | Callable[..., object]
type Leg = tuple[str | Source, str | Sink | None, Amount] | tuple[str | Source, str | Sink | None, Amount, str]
type NamedLeg = tuple[str | Source, str | Sink | None, Amount, str]


class TransferInteraction(Interaction):
    """Transfer of amounts of states from the members of a source role to the members of a target role.

    A leg moves an amount out of a state of the source member, or out of `Source()`, into a state of the target member,
    or into `Sink()`. The transfer has a first leg, `state` into `into` by `amount`, and any further `legs`, each
    `(state, into, amount)` or `(state, into, amount, flow)`. The amounts of all legs are read before any of them
    moves, so a leg sees the states as they were before the transfer. A guard skips the transfer where it does not
    hold.

    The symbolic form reports the amount of every leg as a flow. A leg named by its `flow` is reported under that
    name: under the element or group, as `<path>.<flow>`, when the source and the target are one element or group,
    since the flow then moves a quantity within it like a flow of its own, and under the interaction, as
    `<label>.<flow>`, otherwise. A leg without a name is reported under the interaction as `<label>.flow_<index>`.

    The form chooses how the transfer lowers:

    - `'symbolic'` lowers to `TransferLaw`s between the source and the target of one element, or member `i` of both
      groups. The amounts, the guard, and the rate are numbers or functions of the namespace, and the roles are read as
      `namespace.source` and `namespace.target`. With a `rate`, the amount is the size of a jump at the rate.
    - `'agent'` lowers to an agent rule, whose amounts and guard are numbers or functions of the source agent, the
      target agent, and the model view. The pairing `'members'` pairs member `i` of both groups, `'random'` pairs every
      source member with a random target member, `'pairs'` draws disjoint random pairs, `'neighbors'` pairs every
      source member with the target members in the cells or nodes around it, `'neighbor-pairs'` draws disjoint random
      pairs of neighbours, every source member in random order taking a random unpaired target member around it, and
      `'cell'` pairs the source and target members that stand in one cell or node. The pairs act in random order. With
      a `fallback` group, the members of that group are the targets of a step in which no target member lives. A leg
      end of the agent form may name `'<reference>.<state>'`, the state of the member that a reference state of the
      agent names, such as `'bank.deposits'` for the deposits of a customer's bank, declared
      `StateSpec(reference='banks')`.

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

        flow:
            The name the symbolic form reports the first leg under, or the empty string for `flow_0`.

        pairing:
            How members are paired, one of `'members'`, `'random'`, `'pairs'`, `'neighbors'`, `'neighbor-pairs'`, or
            `'cell'`, the agent form alone taking the others than `'members'`.

        legs:
            Further legs, each a tuple of a state or `Source()`, a state, `Sink()`, or `None`, an amount, and
            optionally the name of its flow.

        fallback:
            The path of the group whose members the agent form pairs with instead of the target members in a step in
            which no target member lives, or `None` for no fallback.

        guard:
            A function that returns whether a pair transfers, a condition of the namespace or a boolean of the agents,
            or `None` for every pair.

        rate:
            The rate of jumps of the symbolic form, a number or a function of the namespace, or `None` for a
            continuous flow.

        form:
            The lowering, `'symbolic'` to transfer laws or `'agent'` to an agent rule.

        phase:
            The phase the rule of the agent form runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The form or the pairing is unknown, a leg is malformed, two legs share a flow name, or the symbolic form has
            an agent pairing or a fallback, or the agent form a rate or a flow name, raised by `check`.

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
        flow: str = '',
        pairing: str = 'members',
        legs: Sequence[Leg] = (),
        fallback: Annotated[str | None, RoleSpec()] = None,
        guard: Callable[..., object] | None = None,
        rate: Amount | None = None,
        form: str = 'symbolic',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.state = state
        self.into = into
        self.amount = amount
        self.flow = flow
        self.pairing = pairing
        self.legs = legs
        self.fallback = fallback
        self.guard = guard
        self.rate = rate
        self.form = form
        self.phase = phase

    def _get_legs(self) -> list[NamedLeg]:
        """Return every leg with the state it adds to resolved and the name of its flow, empty when it has none."""
        legs: list[NamedLeg] = []
        for state, into, amount, *flow in ((self.state, self.into, self.amount, self.flow), *self.legs):
            name = flow[0] if flow else ''
            legs.append((state, state if into is None and isinstance(state, str) else into, amount, name))
        return legs

    def _check_leg(self, state: object, into: object, amount: object, flow: object) -> None:
        """Raise an error for a leg with unusable ends, amount, or flow name, or a symbolic referenced end."""
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
        if not isinstance(flow, str):
            error_msg = f'A leg of {self.label!r} names its flow {flow!r}. Give a string, or the empty string.'
            raise ValidationError(error_msg)
        referenced = [end for end in (state, into) if isinstance(end, str) and '.' in end]
        if referenced and self.form == 'symbolic':
            error_msg = f'{self.label!r} moves {referenced[0]!r} of a referenced member. Use `form="agent"`.'
            raise ValidationError(error_msg)

    def _check_flows(self) -> None:
        """Raise an error for flow names that two legs share, or that the agent form, which reports no flows, takes."""
        names = [leg[3] for leg in self._get_legs() if leg[3]]
        if len(set(names)) < len(names):
            error_msg = f'Two legs of {self.label!r} share a flow name among {names!r}. Give every flow its own name.'
            raise ValidationError(error_msg)
        if names and self.form == 'agent':
            error_msg = f'The agent form of {self.label!r} reports no flows, but it names {names!r}. Remove the names.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown form or pairing, a malformed leg, or an option the form does not take.

        Raises:
            ValidationError:
                The form or the pairing is unknown, a leg is malformed, two legs share a flow name, the symbolic form
                has an agent pairing, a fallback, or a leg end of a referenced member, or the agent form has a rate or
                a flow name.
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
        if self.form == 'symbolic' and self.fallback is not None:
            error_msg = f'The symbolic form of {self.label!r} pairs members. Use `form="agent"` for a fallback group.'
            raise ValidationError(error_msg)
        for leg in self.legs:
            if not isinstance(leg, tuple) or len(leg) not in LEG_LENGTHS:
                error_msg = f'A leg of {self.label!r} is {leg!r}. Give a tuple `(state, into, amount[, flow])`.'
                raise ValidationError(error_msg)
        for state, into, amount, flow in self._get_legs():
            self._check_leg(state, into, amount, flow)
        self._check_flows()
        if self.rate is not None and self.form == 'agent':
            error_msg = f'{self.label!r} has a rate, which only the symbolic form takes. Give `rate=None`.'
            raise ValidationError(error_msg)
        if self.guard is not None and not callable(self.guard):
            error_msg = f'The guard of {self.label!r} is {self.guard!r}. Give a function, or None.'
            raise ValidationError(error_msg)

    def _get_law(self, namespace: SimpleNamespace, leg: NamedLeg) -> TransferLaw:
        """Return the transfer law of one leg between the members of a namespace, reported under its flow name."""
        state, into, amount, flow = leg
        owner = self.source if flow and self.source == self.target else ''
        held = state if isinstance(state, Source) else getattr(namespace.source, state)
        sink = into if isinstance(into, Sink) else getattr(namespace.target, str(into))
        value = amount(namespace) if callable(amount) else amount
        if self.rate is None:
            if self.guard is not None:
                value = where(self.guard(namespace), value, 0)
            return TransferLaw(held, sink, value, label=flow, owner=owner)
        rate = self.rate(namespace) if callable(self.rate) else self.rate
        if self.guard is not None:
            rate = where(self.guard(namespace), rate, 0)
        return TransferLaw(held, sink, value, rate, label=flow, owner=owner)

    def _get_pairs(self, model: ModelViewProtocol) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of source and target agents of one step in random order."""
        sources = model.agents('source')
        targets = sources if self.source == self.target else model.agents('target')
        if self.fallback is not None and not any(target.alive for target in targets):
            targets = model.agents('fallback')
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        if self.pairing == 'members':
            pairs = list(zip(sources, targets, strict=False))
        elif self.pairing == 'random':
            pairs = self._get_random_pairs(sources, targets, model)
        elif self.pairing == 'pairs':
            pairs = self._get_disjoint_pairs(sources, targets, model)
        elif self.pairing == 'neighbor-pairs':
            pairs = self._get_neighbor_pairs(sources, targets, model)
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

    @staticmethod
    def _locate(targets: Sequence[AgentViewProtocol]) -> dict[object, list[AgentViewProtocol]]:
        """Return the target agents by the key of the cell or node they stand in."""
        located: dict[object, list[AgentViewProtocol]] = {}
        for target in targets:
            located.setdefault(target.location_key, []).append(target)
        return located

    def _get_neighbor_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
        model: ModelViewProtocol,
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return disjoint random pairs of neighbours, every source agent in random order taking a free one nearby."""
        located = self._locate(targets)
        paired: set[int] = set()
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        for index in model.rng.permutation(len(sources)):
            source = sources[int(index)]
            if source.id in paired or not source.alive:
                continue
            free = [
                target
                for key in source.neighbor_cells()
                for target in located.get(key, [])
                if target.id != source.id and target.id not in paired and target.alive
            ]
            if free:
                target = free[int(model.rng.integers(len(free)))]
                pairs.append((source, target))
                paired.update((source.id, target.id))
        return pairs

    def _get_located_pairs(
        self,
        sources: Sequence[AgentViewProtocol],
        targets: Sequence[AgentViewProtocol],
    ) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of agents in one cell, or of every source agent with the target agents around it."""
        located = self._locate(targets)
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        for source in sources:
            keys = [source.location_key] if self.pairing == 'cell' else source.neighbor_cells()
            pairs.extend((source, target) for key in keys for target in located.get(key, []) if target is not source)
        return pairs

    def _evaluate(
        self,
        amount: Amount,
        source: AgentViewProtocol,
        target: AgentViewProtocol,
        model: ModelViewProtocol,
    ) -> float:
        """Return the amount of a leg between two agents as a number."""
        value = amount(source, target, model) if callable(amount) else amount
        if not isinstance(value, int | float | np.floating | np.integer):
            error_msg = f'An amount of {self.label!r} between agents is {value!r}. Return a number.'
            raise ValidationError(error_msg)
        return float(value)

    def _transfer_pairs(self, model: ModelViewProtocol) -> None:
        """Transfer between every pair of agents of one step."""
        for source, target in self._get_pairs(model):
            if source.alive and target.alive:
                self.transfer(source, target, model)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer laws of the symbolic form, or the rule of the agent form.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                A transfer law per leg, or the rule that transfers between the pairs of agents.
        """
        if self.form == 'agent':
            return [RuleLaw(self._transfer_pairs, phase=self.phase)]
        return [self._get_law(namespace, leg) for leg in self._get_legs()]

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Move the amounts of every leg from one source agent to one target agent, when the guard holds.

        Args:
            source:
                The agent the amounts leave.

            target:
                The agent the amounts enter.

            model:
                The model view of the rule.

        Raises:
            ValidationError:
                An amount function returns something other than a number.

            SimulationError:
                A leg end names the state of a referenced member, and the reference state names no member.
        """
        if self.guard is not None and not self.guard(source, target, model):
            return
        legs = self._get_legs()
        amounts = [self._evaluate(amount, source, target, model) for _, _, amount, _ in legs]
        ends = [
            (
                self._get_end(source, state) if isinstance(state, str) else None,
                self._get_end(target, into) if isinstance(into, str) else None,
            )
            for state, into, _, _ in legs
        ]
        for (taken, given), amount in zip(ends, amounts, strict=True):
            if taken is not None:
                holder, name = taken
                setattr(holder, name, getattr(holder, name) - amount)
            if given is not None:
                holder, name = given
                setattr(holder, name, getattr(holder, name) + amount)

    def _get_end(self, agent: AgentViewProtocol, name: str) -> tuple[AgentViewProtocol, str]:
        """Return the agent that holds the state a leg end names and the name of the state.

        A name `'<reference>.<state>'` names the state of the member that the reference state of the agent names.
        """
        reference, _, state = name.rpartition('.')
        if not reference:
            return agent, name
        holder = agent.referenced(reference)
        if holder is None:
            error_msg = (
                f'A leg of {self.label!r} moves {name!r}, but the state {reference!r} of agent {agent.id} names no '
                'member.'
            )
            raise SimulationError(error_msg)
        return holder, state
