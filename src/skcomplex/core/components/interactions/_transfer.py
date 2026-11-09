"""Define transfers.

Transfers move amounts of states between members, the cells agents stand on, sources, and sinks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...errors import SimulationError, ValidationError
from ...specs import PartnerSpec, RoleSpec, Tags
from ...types import ElementKind
from ..laws import BaseLaw, MatchLaw, RuleLaw, Sink, Source, TransferLaw
from ..laws.utils import shift_member, where
from ..spaces import BaseSpace
from ..types import AgentViewProtocol, Pairing, SystemViewProtocol
from ._interaction import Interaction

FORMS = ('symbolic', 'agent')
PAIRINGS = ('members', 'random', 'pairs', 'neighbors', 'neighbor-pairs', 'cell', 'site')
GRID_PAIRINGS = ('neighbors', 'neighbor-pairs', 'cell', 'site')
SCHEDULES = ('random-sequential', 'member')
ACTORS = ('source', 'target')
REMOVALS = ('source', 'target')
LEG_LENGTHS = (3, 4)
BOUNDARIES = ('periodic', 'fixed')
NAMED_PARTNERS = {
    'random': PartnerSpec('all', 'uniform'),
    'pairs': PartnerSpec('all', 'uniform', capacity=1),
    'neighbors': PartnerSpec('neighbors', 'all'),
    'neighbor-pairs': PartnerSpec('neighbors', 'uniform', capacity=1),
    'cell': PartnerSpec('cell', 'all'),
}

type Amount = float | Callable[..., object]
type Leg = tuple[str | Source, str | Sink | None, Amount] | tuple[str | Source, str | Sink | None, Amount, str]
type NamedLeg = tuple[str | Source, str | Sink | None, Amount, str]
type Settlement = Callable[..., None]


class TransferInteraction(Interaction):
    """Transfer of amounts of states from source-role members to target-role members.

    A leg moves an amount out of a state of the source member, or out of `Source()`, into a state of the target member,
    or into `Sink()`. The transfer has a first leg, `state` into `into` by `amount`, and any further `legs`, each
    `(state, into, amount)` or `(state, into, amount, flow)`. The amounts of all legs are read before any of them
    moves, so a leg sees the states as they were before the transfer. A guard skips the transfer where it does not
    hold.

    The symbolic form reports the amount of every leg as a flow. A leg named by its `flow` is reported under that
    name: under the element or group, as `<path>.<flow>`, when the source and the target are one element or group,
    since the flow then moves a quantity within it like a flow of its own, and under the interaction, as
    `<label>.<flow>`, otherwise, as it is for a subclass that names its roles otherwise. A leg without a name is
    reported under the interaction as `<label>.flow_<index>`.

    The form chooses how the transfer lowers:

    - `'symbolic'` lowers to `TransferLaw`s between the source and the target of one element, or member `i` of both
      groups. The amounts, the guard, and the rates are numbers or functions of the namespace, and the roles are read
      as `namespace.source` and `namespace.target`. With a `rate`, the amount is the size of a jump at the rate, and
      with a `reverse_rate` jumps also move the amounts back from the target to the source at that rate, so a leg from
      `Source()` returns its amount to the environment. With an `offset`, the source is a group and every member
      transfers along an edge of the group order, to the member `offset` places along it, which `namespace.target`
      reads. Past the ends of a `'fixed'` boundary no member is there, and the edge moves nothing. A draw in an amount
      is one draw per edge, which both of its ends read, so the amounts of a step move in parallel. A leg end
      `'<role>.<state>'` names the state of the element or group another role binds, so one transfer moves amounts
      between several holders, such as the payments of a circuit of sectors, and a subclass that declares parameters
      reads them in its amounts. A subclass that names its roles otherwise returns their names from `_get_role`, or
      the namespaces of the source and the target of every pair from `_get_holders`.
    - `'agent'` lowers to an agent rule, whose amounts and guard are numbers or functions of the source agent, the
      target agent, and the system view. The pairing `'members'` pairs member `i` of both groups, `'random'` pairs every
      source member with a random target member, `'pairs'` draws disjoint random pairs, `'neighbors'` pairs every
      source member with the target members in the cells or nodes around it, `'neighbor-pairs'` draws disjoint random
      pairs of neighbours, every source member in random order taking a random unpaired target member around it, and
      `'cell'` pairs the source and target members that stand in one cell or node. With a `fallback` group, the members
      of that group are the targets of a step in which no target member lives. With the pairing `'site'` the target
      role binds a field, and every source member moves its amounts into the site of the field at its cell or node, so
      a negative amount takes from the site. The functions of this pairing take the cell of the source agent in place
      of a target agent. With `remove`, the source or the target member of every pair that transfers is removed, as an
      agent eaten whole. The named pairings other than `'members'` and `'site'` are shorthands of a `PartnerSpec`, which
      pairs every source member with partners within a radius, by score, or with a fallback. A `settlement` receives
      both partners and the amounts of the legs, read once, and changes the states in place of the legs, so one draw
      credits both ends. A subclass that names its roles otherwise returns their names from `_get_role`.

    Under the schedule `'random-sequential'` the pairs act in random order. Under `'member'` every member of the
    `actor` role acts in its own activation, together with the rules of its element and the other member
    interactions, in the order the system lists them: it transfers with its partners, the members of the other role
    that the pairing `'random'`, `'neighbors'`, or `'cell'` gives it, or with `partners` of them drawn at random, and
    with itself under the pairing `'members'` within one group.
    A leg end of the agent form may name `'<reference>.<state>'`, the state of the member that a reference state of
    the agent names, such as `'bank.deposits'` for the deposits of a customer's bank, declared
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
            How members are paired, one of `'members'`, `'random'`, `'pairs'`, `'neighbors'`, `'neighbor-pairs'`,
            `'cell'`, or `'site'`, the agent form alone taking the others than `'members'`, or, in the agent form, a
            `PartnerSpec`, or a function `pairing(sources, targets, rng, capacity)` of the source and target agents
            that returns their pairs, whose capacity is `partners`, or 1 without it. Under the schedule `'member'` the
            spec or the function receives the actor alone as its first sequence and the actor's partners are the second
            items of its pairs.

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

        reverse_rate:
            The rate of the jumps of the symbolic form that move the amounts back from the target to the source, a
            number or a function of the namespace, or `None` for no reverse jumps.

        offset:
            The number of places along the group order from every source member to its target member, or 0 to pair
            the members of the roles.

        boundary:
            Past the ends of the group order, `'periodic'` closes a ring and `'fixed'` holds no member.

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

        settlement:
            A function `settlement(source, target, amounts, system)` that settles a pair of agents from the amounts of
            the legs, read once, in place of moving them, or `None` to move them.

    Raises:
        ValidationError:
            The form, the pairing, the schedule, the actor, the removal, or the boundary is unknown, a leg is
            malformed, two legs share a flow name, the symbolic form has an agent pairing, a fallback, or a settlement,
            the agent form a rate, an offset, or a flow name, a reverse rate has no rate, an offset is not a whole
            number or pairs two groups, a leg end of the symbolic form names no role, the settlement cannot be called
            or pairs with a site, or the schedule `'member'` has a pairing it cannot draw, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import TransferInteraction
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.specs import StateSpec
        >>> from typing import Annotated
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class TankElement(Element):
        ...     def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.level = level
        >>> flow = TransferInteraction(
        ...     'flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2
        ... )
        >>> system = System(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
        >>> trajectory = simulate(system, until=1.0)
        >>> round(float(trajectory.values('upper.level')[-1] + trajectory.values('lower.level')[-1]), 6)
        4.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (MatchLaw, RuleLaw, TransferLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the rules of the agent form or of the transfers.

        The transfers jump when they have a rate.
        """
        if self._get_argument('form') == 'agent':
            return self._get_variant_tags((RuleLaw,))
        return self._get_variant_tags((TransferLaw,), stochastic=self._get_argument('rate') is not None)

    def _draws(self) -> bool:
        """Return whether the agent form draws.

        It draws by its order, random pairs or partners, or a number of partners.
        """
        if self._get_argument('form') != 'agent':
            return self.draws
        drawn = self.pairing in ('random', 'pairs', 'neighbor-pairs') or not isinstance(self.pairing, str | None)
        return super()._draws() or drawn or self.partners is not None

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec(elements=(ElementKind.ANY,))],
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,))],
        state: str | Source,
        into: str | Sink | None = None,
        amount: Amount,
        flow: str = '',
        pairing: str | Pairing | PartnerSpec = 'members',
        legs: Sequence[Leg] = (),
        fallback: Annotated[str | None, RoleSpec(elements=(ElementKind.ANY,))] = None,
        guard: Callable[..., object] | None = None,
        rate: Amount | None = None,
        reverse_rate: Amount | None = None,
        offset: int = 0,
        boundary: str = 'periodic',
        form: str = 'symbolic',
        schedule: str = 'random-sequential',
        actor: str = 'source',
        partners: int | None = None,
        remove: str | None = None,
        phase: str | None = None,
        settlement: Settlement | None = None,
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
        self.reverse_rate = reverse_rate
        self.offset = offset
        self.boundary = boundary
        self.form = form
        self.schedule = schedule
        self.actor = actor
        self.partners = partners
        self.remove = remove
        self.phase = phase
        self.settlement = settlement

    def _get_legs(self) -> list[NamedLeg]:
        """Return every leg with its resolved state and the name of its flow.

        The resolved state is the one it adds to, and the flow name is empty when it has none.
        """
        legs: list[NamedLeg] = []
        for state, into, amount, *flow in ((self.state, self.into, self.amount, self.flow), *self.legs):
            name = flow[0] if flow else ''
            legs.append((state, state if into is None and isinstance(state, str) else into, amount, name))
        return legs

    def _check_leg(self, state: object, into: object, amount: object, flow: object) -> None:
        """Raise an error for a leg with bad ends, amount, or flow name.

        It also raises it for a symbolic end that names no role.
        """
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
        roles = [name for name, spec in self._get_specs().items() if isinstance(spec, RoleSpec)]
        referenced = [end for end in (state, into) if isinstance(end, str) and '.' in end]
        unknown = [end for end in referenced if end.partition('.')[0] not in roles]
        if unknown and self.form == 'symbolic':
            error_msg = (
                f'{self.label!r} moves {unknown[0]!r}, but its symbolic form names the states of roles, one of '
                f'{", ".join(roles)}. Name a role, or use `form="agent"` for a referenced member.'
            )
            raise ValidationError(error_msg)

    def _check_flows(self) -> None:
        """Raise an error for flow names that two legs share, or that the agent form takes.

        The agent form reports no flows.
        """
        names = [leg[3] for leg in self._get_legs() if leg[3]]
        if len(set(names)) < len(names):
            error_msg = f'Two legs of {self.label!r} share a flow name among {names!r}. Give every flow its own name.'
            raise ValidationError(error_msg)
        if names and self.form == 'agent':
            error_msg = f'The agent form of {self.label!r} reports no flows, but it names {names!r}. Remove the names.'
            raise ValidationError(error_msg)

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for a pairing of cells, nodes, or sites in a continuous space."""
        if isinstance(self.pairing, str) and self.pairing in GRID_PAIRINGS:
            alternative = "a radius pairing, `PartnerSpec(candidates=<radius>)`, or pairing='random'"
            self._check_grid('pairing', self.pairing, space, alternative)
        self._check_partners(self.pairing, space)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown form or pairing, a bad leg, or a bad option.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The form, the pairing, the schedule, the actor, the removal, or the boundary is unknown, a leg is
                malformed, two legs share a flow name, the symbolic form has an agent pairing, a fallback, or a leg end
                that names no role, the agent form has a rate, an offset, or a flow name, a reverse rate has no rate,
                an offset is not a whole number or pairs two groups, or the schedule `'member'` has a pairing it
                cannot draw.
        """
        super()._prepare(context)
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.pairing != 'members':
            error_msg = f'The symbolic form of {self.label!r} pairs members. Use `form="agent"` for {self.pairing!r}.'
            raise ValidationError(error_msg)
        self._check_variant('pairing', self.pairing, PAIRINGS, (PartnerSpec,), functions=True)
        if isinstance(self.pairing, PartnerSpec):
            self.pairing._check()
        self._check_settlement()
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
        self._check_edges()
        self._check_schedule()

    def _check_settlement(self) -> None:
        """Raise an error for an uncallable settlement, or one its form or pairing rejects."""
        if self.settlement is None:
            return
        if not callable(self.settlement):
            error_msg = f'The settlement of {self.label!r} is {self.settlement!r}. Give a function, or None.'
            raise ValidationError(error_msg)
        if self.form != 'agent' or self.pairing == 'site':
            error_msg = f'{self.label!r} settles a pair of agents. Use `form="agent"` and a pairing other than `site`.'
            raise ValidationError(error_msg)

    def _check_edges(self) -> None:
        """Raise an error for an unusable reverse rate, offset, or boundary."""
        if self.reverse_rate is not None and self.rate is None:
            error_msg = f'{self.label!r} has a reverse rate but no rate. Give the rate of the forward jumps.'
            raise ValidationError(error_msg)
        reverse = self.reverse_rate
        if reverse is not None and not callable(reverse) and not isinstance(reverse, int | float):
            error_msg = f'The reverse rate {reverse!r} of {self.label!r} is neither a number nor a function.'
            raise ValidationError(error_msg)
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            error_msg = f'The offset of {self.label!r} is {self.offset!r}. Give a whole number of members.'
            raise ValidationError(error_msg)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown boundary {self.boundary!r} of {self.label!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if self.offset and (self.form != 'symbolic' or self.source != self.target):
            error_msg = (
                f'{self.label!r} transfers along the order of one group by an offset. Give the group as both source '
                'and target, and `form="symbolic"`.'
            )
            raise ValidationError(error_msg)

    def _check_schedule(self) -> None:
        """Raise an error for an unknown schedule, actor, or removal, or bad partners.

        The schedule cannot draw the bad partners.
        """
        for name, value, variants in (
            ('schedule', self.schedule, SCHEDULES),
            ('actor', self.actor, ACTORS),
            ('removal', self.remove, (*REMOVALS, None)),
        ):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {", ".join(map(str, variants))}.'
                raise ValidationError(error_msg)
        own = self.pairing == 'members' and self.source == self.target
        if self.schedule == 'member' and self.pairing in {'members', 'pairs', 'neighbor-pairs'} and not own:
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
        if isinstance(self.pairing, PartnerSpec) and self.schedule == 'member' and self.pairing.capacity is not None:
            error_msg = (
                f'The partners of {self.label!r} have a capacity of one step, but under the schedule `member` every '
                'actor draws alone. Use the schedule `random-sequential`, or no capacity.'
            )
            raise ValidationError(error_msg)
        if isinstance(self.pairing, PartnerSpec) and self.partners is not None:
            error_msg = f'{self.label!r} draws partners by a `PartnerSpec`. Give its capacity, not `partners`.'
            raise ValidationError(error_msg)
        if self.partners is not None and (isinstance(self.partners, bool) or not self.partners >= 1):
            error_msg = f'The partners of {self.label!r} are {self.partners!r}. Give a positive whole number, or None.'
            raise ValidationError(error_msg)

    def _get_owner(self) -> str:
        """Return the path of the element or group both source and target roles bind.

        It returns the empty string otherwise.
        """
        roles = self._get_specs()
        bound = all(isinstance(roles.get(self._get_role(role)), RoleSpec) for role in ('source', 'target'))
        return self.source if bound and self.source == self.target else ''

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the namespaces of the source and the target of every pair.

        These are the pairs the symbolic form transfers between.
        """
        return [(getattr(namespace, self._get_role('source')), getattr(namespace, self._get_role('target')))]

    def _shift(self, holder: SimpleNamespace) -> SimpleNamespace:
        """Return the namespace of a group with every symbol read at an offset member.

        The offset names the member.
        """
        if not self.offset:
            return holder
        return SimpleNamespace(
            **{
                name: (
                    shift_member(value, self.offset, boundary=self.boundary) if isinstance(value, Expression) else value
                )
                for name, value in vars(holder).items()
            },
        )

    @staticmethod
    def _get_holder(namespace: SimpleNamespace, end: str, own: SimpleNamespace) -> object:
        """Return the symbol of a leg end.

        It is the state of the role it names or the state of its own side.
        """
        role, _, state = end.rpartition('.')
        return getattr(getattr(namespace, role) if role else own, state)

    def _get_law(
        self,
        namespace: SimpleNamespace,
        leg: NamedLeg,
        rate: Amount | None,
        reverse: bool,
    ) -> TransferLaw:
        """Return the transfer law of one leg of a pair, forward or back.

        It is reported under its flow name.
        """
        state, into, amount, flow = leg
        owner = self._get_owner() if flow else ''
        held = (
            state
            if isinstance(state, Source)
            else self._get_holder(namespace, state, self._get_member(namespace, 'source'))
        )
        sink = (
            into
            if isinstance(into, Sink)
            else self._get_holder(namespace, str(into), self._get_member(namespace, 'target'))
        )
        value = amount(namespace) if callable(amount) else amount
        if reverse:
            held, sink = (
                Source(sink.label) if isinstance(sink, Sink) else sink,
                Sink(held.label) if isinstance(held, Source) else held,
            )
            flow = f'{flow}_reverse' if flow else ''
        if rate is None:
            if self.guard is not None:
                value = where(self.guard(namespace), value, 0)
            return TransferLaw(held, sink, value, label=flow, owner=owner)
        jumps = rate(namespace) if callable(rate) else rate
        if self.guard is not None:
            jumps = where(self.guard(namespace), jumps, 0)
        return TransferLaw(held, sink, value, jumps, label=flow, owner=owner)

    def _get_transfer_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfer laws of every leg of every pair of the symbolic form.

        They go forward and back.
        """
        laws: list[BaseLaw] = []
        for source, target in self._get_holders(namespace):
            paired = SimpleNamespace(
                **{**vars(namespace), self._get_role('source'): source, self._get_role('target'): self._shift(target)},
            )
            laws += [self._get_law(paired, leg, self.rate, False) for leg in self._get_legs()]
            if self.reverse_rate is not None:
                laws += [self._get_law(paired, leg, self.reverse_rate, True) for leg in self._get_legs()]
        return laws

    def _get_pairing(self) -> MatchLaw:
        """Return the law of the pairs of the pairing.

        Its capacity for a function is `partners`, or 1 without it.
        """
        pairing = self.pairing
        spec = NAMED_PARTNERS[pairing] if isinstance(pairing, str) else pairing
        return MatchLaw(self.source, self.target, spec, capacity=self.partners or 1)

    def _get_pairs(self, system: SystemViewProtocol) -> list[tuple[AgentViewProtocol, AgentViewProtocol]]:
        """Return the pairs of source and target agents of one step in random order."""
        sources = system.agents(self._get_role('source'))
        targets = sources if self.source == self.target else system.agents(self._get_role('target'))
        if self.fallback is not None and not any(target.alive for target in targets):
            targets = system.agents(self._get_role('fallback'))
        pairs: list[tuple[AgentViewProtocol, AgentViewProtocol]] = []
        if self.pairing == 'members':
            pairs = list(zip(sources, targets, strict=False))
        else:
            pairs = cast(
                list[tuple[AgentViewProtocol, AgentViewProtocol]],
                self._get_pairing().get_pairs(sources, targets, system.rng),
            )
        return [pairs[int(index)] for index in system.rng.permutation(len(pairs))]

    def _evaluate(
        self,
        amount: Amount,
        source: AgentViewProtocol,
        target: object,
        system: SystemViewProtocol,
    ) -> float:
        """Return the amount of a leg between two agents as a number."""
        value = amount(source, target, system) if callable(amount) else amount
        if not isinstance(value, int | float | np.floating | np.integer):
            error_msg = f'An amount of {self.label!r} between agents is {value!r}. Return a number.'
            raise ValidationError(error_msg)
        return float(value)

    def _transfer_pair(self, source: AgentViewProtocol, target: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Transfer between two living agents and remove the member a removal names."""
        if source.alive and target.alive and self.transfer(source, target, system) and self.remove is not None:
            (source if self.remove == 'source' else target).remove()

    def _deposit(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move the amounts of every leg from an agent into the site of the target field.

        The site is at its cell or node.
        """
        cell = agent.location_key
        if not agent.alive or (self.guard is not None and not self.guard(agent, cell, system)):
            return
        legs = self._get_legs()
        amounts = [self._evaluate(amount, agent, cell, system) for _, _, amount, _ in legs]
        field = self._get_role('target')
        for (state, into, _, _), amount in zip(legs, amounts, strict=True):
            if isinstance(state, str):
                setattr(agent, state, getattr(agent, state) - amount)
            if isinstance(into, str):
                system.deposit(f'{field}.{into}', cell, amount)

    def _get_partners(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> list[AgentViewProtocol]:
        """Return the partners of an actor by the pairing, or a random number of them."""
        if self.pairing == 'members':
            return [agent]
        role = self._get_role('source' if self.actor == 'target' else 'target')
        if callable(self.pairing) or isinstance(self.pairing, PartnerSpec):
            others = [other for other in system.agents(role) if other is not agent]
            pairs = self._get_pairing().get_pairs([agent], others, system.rng)
            return [cast(AgentViewProtocol, partner) for _, partner in pairs]
        if self.pairing == 'cell':
            groups = set(system.groups(role))
            others = [other for other in agent.agents_here() if other.group in groups]
        else:
            others = [other for other in system.agents(role) if other is not agent]
        if self.pairing == 'neighbors':
            cells = set(agent.neighbor_cells())
            others = [other for other in others if other.location_key in cells]
        elif self.pairing == 'random' and others:
            others = [others[int(system.rng.integers(len(others)))]]
        if self.partners is None or len(others) <= self.partners:
            return others
        return [others[int(index)] for index in system.rng.permutation(len(others))[: self.partners]]

    def _transfer_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Transfer between one actor and its partners, in the activation of the actor."""
        if self.pairing == 'site':
            self._deposit(agent, system)
            return
        for partner in self._get_partners(agent, system):
            if agent.alive:
                pair = (partner, agent) if self.actor == 'target' else (agent, partner)
                self._transfer_pair(*pair, system)

    def _transfer_pairs(self, system: SystemViewProtocol) -> None:
        """Transfer between every pair of agents of one step.

        The transfer may go into the sites of the source agents instead.
        """
        if self.pairing == 'site':
            sources = system.agents(self._get_role('source'))
            for index in system.rng.permutation(len(sources)):
                self._deposit(sources[int(index)], system)
            return
        for source, target in self._get_pairs(system):
            self._transfer_pair(source, target, system)

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
            return [RuleLaw(self._transfer_member, phase=self.phase, members=self._get_role(self.actor))]
        if self.form == 'agent':
            return [RuleLaw(self._transfer_pairs, phase=self.phase)]
        return self._get_transfer_laws(namespace)

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Move the amounts of every leg from one source agent to one target agent.

        It moves them when the guard holds.

        Args:
            source:
                The agent the amounts leave.

            target:
                The agent the amounts enter.

            system:
                The system view of the rule.

        Returns:
            transferred:
                Whether the guard held, so the amounts moved.

        Raises:
            ValidationError:
                An amount function returns something other than a number.

            SimulationError:
                A leg end names the state of a referenced member, and the reference state names no member.
        """
        if self.guard is not None and not self.guard(source, target, system):
            return False
        legs = self._get_legs()
        amounts = [self._evaluate(amount, source, target, system) for _, _, amount, _ in legs]
        if self.settlement is not None:
            self.settlement(source, target, amounts, system)
            return True
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
        return True

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
