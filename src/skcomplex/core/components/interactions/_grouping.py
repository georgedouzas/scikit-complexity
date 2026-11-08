"""Define groupings.

A grouping draws groups of the members of a group every step and acts on each group.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import BaseSpec, PartnerSpec, ReductionSpec, RoleSpec, Tags
from ...types import ElementKind
from ..laws import BaseLaw, MatchLaw, RuleLaw, UpdateLaw
from ..laws.utils import both, count, maximum, mean, minimum, reductions_in, total, variance
from ..spaces import BaseSpace
from ..types import AgentViewProtocol, Pairing, SystemViewProtocol
from ._interaction import Interaction

GROUPS = ('pairs', 'partition', 'neighborhood', 'state')
FORMS = ('agent', 'symbolic')
REDUCES = ('sum', 'mean')
PAIR = 2
KEYED: dict[str, Callable[..., Expression]] = {
    'sum': total,
    'mean': mean,
    'max': maximum,
    'min': minimum,
    'var': variance,
}

type Act = Callable[[list[AgentViewProtocol], SystemViewProtocol], None]
type GroupFunction = Callable[[SimpleNamespace], object]


class GroupingInteraction(Interaction):
    """Grouping of the members of a group into groups drawn every step.

    An action applies to every group.

    The groups are one of four kinds:

    - `'pairs'` draws disjoint random pairs, and an odd member out forms no group.
    - `'partition'` draws a random partition into groups of `size` members, the last group holding the remainder.
    - `'neighborhood'` forms one group per member, the member first and then the members in the cells or nodes
      around it.
    - `'state'` forms one group per value of the state `by`, of at most `capacity` agents drawn at random among those
      that hold the value in the agent form.

    In the agent form the groups act in random order, each through `act(members, system)`, a function of the list of
    agent views and the system view, or the method `act_on` a subclass overrides. The grouping is an agent rule, so the
    system runs in discrete time.

    In the symbolic form every member of a group takes the value of `function(namespace)` for the state `write` in the
    next step, reduced by `reduce` over the groups that hold it, and a member in no group keeps its value. The
    namespace of a group holds `member`, the member the value is for, `others`, the other members in the order of the
    group, `members`, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with `over=namespace.members`,
    the time `t`, and the parameters of the interaction. Pairs and partitions are drawn again at every step from the
    random stream of the simulation, and neighbourhoods are those of the space. The groups by state hold no `others`,
    and their reductions over the members are keyed by the state. Within a step, the grouping runs in its `phase` and
    reads the states the phases before it set.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group whose members are grouped.

        groups:
            The kind of groups, `'pairs'`, `'partition'`, `'neighborhood'`, or `'state'`, or in the agent form a
            `PartnerSpec` or a function `pairing(members, members, rng, capacity)` that returns pairs, which group
            every member with its partners. `'pairs'` is a shorthand of `PartnerSpec('all', 'uniform', capacity=1)`.

        act:
            The function of the members of a group and the system view that acts on the group in the agent form, used
            when a subclass does not override `act_on`.

        size:
            The number of members of a group of a partition.

        by:
            The name of the state whose values the groups `'state'` share.

        capacity:
            The largest number of agents of a group `'state'` in the agent form, or `None` for no limit.

        phase:
            The phase of the step the grouping runs in, or `None` for the default phase.

        form:
            `'agent'` for an action on the agents of every group, or `'symbolic'` for a function of every group.

        write:
            The name of the state that the symbolic form sets.

        function:
            The function of the namespace of one group that returns the value of `write` for its member, in the
            symbolic form.

        reduce:
            `'sum'` or `'mean'`, how the symbolic form combines the values of the groups that hold a member.

    Raises:
        ValidationError:
            The kind of groups, the form, or the reduction is unknown, the size is below two, the groups `'state'` have
            no state, the capacity is not positive, the symbolic form has no state or function, or a setting of one
            form is given to the other, raised by `prepare`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.interactions import GroupingInteraction
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.specs import StateSpec
        >>> class VoterElement(Element):
        ...     def __init__(self, label, visual=None, *, s: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.s = s
        >>> def adopt_majority(members, system):
        ...     majority = float(2 * sum(member.s for member in members) > len(members))
        ...     for member in members:
        ...         member.s = majority
        >>> voters = GroupElement('voters', element=VoterElement, size=3, values={'s': [1.0, 1.0, 0.0]})
        >>> galam = GroupingInteraction('galam', target='voters', groups='partition', size=3, act=adopt_majority)
        >>> trajectory = simulate(System(elements=[voters], interactions=[galam]), until=1)
        >>> trajectory.agents('voters', 1).values['s'].tolist()
        [1.0, 1.0, 1.0]

        In the symbolic form every group of the partition takes the opinion of its majority:

        >>> from skcomplex.core.components.laws.utils import mean, where
        >>> galam = GroupingInteraction(
        ...     'galam',
        ...     target='voters',
        ...     groups='partition',
        ...     size=3,
        ...     form='symbolic',
        ...     write='s',
        ...     function=lambda g: where(mean(g.members.s, over=g.members) > 1 / 2, 1, 0),
        ... )
        >>> trajectory = simulate(System(elements=[voters], interactions=[galam]), until=1, random_state=0)
        >>> trajectory.values('voters.s')[-1].tolist()
        [1.0, 1.0, 1.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (MatchLaw, RuleLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of the agent-form rules or of the symbolic grouping updates."""
        return self._get_variant_tags((RuleLaw,) if self._get_argument('form') == 'agent' else (UpdateLaw,))

    def _draws(self) -> bool:
        """Return whether the grouping draws.

        It draws by random groups or by acting on its groups in random order.
        """
        return super()._draws() or self._get_argument('form') == 'agent' or self.groups != 'neighborhood'

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('by', 'write'))],
        groups: str = 'pairs',
        act: Act | None = None,
        size: int = 2,
        by: str | None = None,
        capacity: int | None = None,
        phase: str | None = None,
        form: str = 'agent',
        write: str | None = None,
        function: GroupFunction | None = None,
        reduce: str = 'sum',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.groups = groups
        self.act = act
        self.size = size
        self.by = by
        self.capacity = capacity
        self.phase = phase
        self.form = form
        self.write = write
        self.function = function
        self.reduce = reduce

    def _check_symbolic(self) -> None:
        """Raise an error for a symbolic form without its parts or with agent settings.

        Raises:
            ValidationError:
                The reduction is unknown, the state or the function is missing, or `act` or `capacity` is given.
        """
        if self.reduce not in REDUCES:
            error_msg = f'Unknown reduction {self.reduce!r} of {self.label!r}. Use one of {", ".join(REDUCES)}.'
            raise ValidationError(error_msg)
        for name, value in (('act', self.act), ('capacity', self.capacity)):
            if value is not None:
                error_msg = (
                    f'The grouping {self.label!r} is symbolic, but it was given `{name}`, a setting of the agent form. '
                    f'Remove `{name}`, or use form `agent`.'
                )
                raise ValidationError(error_msg)
        if not isinstance(self.write, str) or not callable(self.function):
            error_msg = (
                f'The symbolic grouping {self.label!r} sets a state to a function of every group. Pass the name of the '
                'state as `write` and the function as `function`.'
            )
            raise ValidationError(error_msg)

    def _check_space(self, space: BaseSpace | None) -> None:
        """Raise an error for groups of neighbouring cells or nodes in a continuous space."""
        if self.groups == 'neighborhood':
            self._check_grid('groups', self.groups, space, "groups='pairs', 'partition', 'state', or a radius")
        self._check_partners(self.groups, space)

    def _check(self) -> None:
        """Raise an error for an unknown kind, a small size, no state, or a small capacity.

        Raises:
            ValidationError:
                The kind of groups, the form, or the reduction is unknown, the size is below two, the groups `'state'`
                have no state, the capacity is not positive, the symbolic form has no state or function, or a setting
                of one form is given to the other.
        """
        super()._check()
        if self.form not in FORMS:
            error_msg = f'Unknown form {self.form!r} of {self.label!r}. Use one of {", ".join(FORMS)}.'
            raise ValidationError(error_msg)
        self._check_variant('groups', self.groups, GROUPS, (BaseSpec,), functions=True)
        if isinstance(self.groups, PartnerSpec):
            self.groups._validated()
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < PAIR:
            error_msg = f'The size of the groups of {self.label!r} is {self.size!r}. Give a whole number of at least 2.'
            raise ValidationError(error_msg)
        if self.groups == 'state' and not isinstance(self.by, str):
            error_msg = f'The groups `state` of {self.label!r} share the value of a state. Pass its name as `by`.'
            raise ValidationError(error_msg)
        if self.capacity is not None and (isinstance(self.capacity, bool) or self.capacity < 1):
            error_msg = f'The capacity of {self.label!r} is {self.capacity!r}. Give a positive whole number, or None.'
            raise ValidationError(error_msg)
        if self.act is not None and not callable(self.act):
            error_msg = f'The action of {self.label!r} is {self.act!r}. Give a function, or None.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic':
            if not isinstance(self.groups, str):
                error_msg = (
                    f'The symbolic form of {self.label!r} compiles the named groups, so it cannot run {self.groups!r}. '
                    'Use `form="agent"`, whose groups `_get_groups` draws.'
                )
                raise ValidationError(error_msg)
            self._check_symbolic()
            return
        if self.write is not None or self.function is not None:
            error_msg = (
                f'The grouping {self.label!r} acts on agents, but it was given `write` or `function`, settings of the '
                'symbolic form. Remove them, or use form `symbolic`.'
            )
            raise ValidationError(error_msg)

    def _get_partition(
        self,
        agents: Sequence[AgentViewProtocol],
        size: int,
        system: SystemViewProtocol,
    ) -> list[list[AgentViewProtocol]]:
        """Return a random partition of agents into groups of a size.

        The remainder goes in the last group.
        """
        order = [agents[int(index)] for index in system.rng.permutation(len(agents))]
        return [order[start : start + size] for start in range(0, len(order), size)]

    def _get_neighborhoods(self, agents: Sequence[AgentViewProtocol]) -> list[list[AgentViewProtocol]]:
        """Return every agent followed by the agents in the cells or nodes around it."""
        located: dict[object, list[AgentViewProtocol]] = {}
        for agent in agents:
            located.setdefault(agent.location_key, []).append(agent)
        return [
            [agent, *(other for cell in agent.neighbor_cells() for other in located.get(cell, []))] for agent in agents
        ]

    def _get_shared(
        self,
        agents: Sequence[AgentViewProtocol],
        system: SystemViewProtocol,
    ) -> list[list[AgentViewProtocol]]:
        """Return the groups of agents that share a value of the state `by`.

        Each group is cut at random to the capacity.
        """
        shared: dict[float, list[AgentViewProtocol]] = {}
        for agent in agents:
            shared.setdefault(float(getattr(agent, str(self.by))), []).append(agent)
        groups = []
        for value in sorted(shared):
            members = shared[value]
            order = [members[int(index)] for index in system.rng.permutation(len(members))]
            groups.append(order if self.capacity is None else order[: self.capacity])
        return groups

    def _get_partners(
        self,
        pairing: PartnerSpec | Pairing,
        agents: Sequence[AgentViewProtocol],
        system: SystemViewProtocol,
    ) -> list[list[AgentViewProtocol]]:
        """Return the groups of every agent with the partners it gets.

        A selection of partners or a function gives the partners.
        """
        law = MatchLaw(self.target, pairing=pairing, capacity=self.capacity or 1)
        groups: dict[int, list[AgentViewProtocol]] = {}
        for first, second in law.get_pairs(agents, agents, system.rng):
            member, partner = cast(AgentViewProtocol, first), cast(AgentViewProtocol, second)
            groups.setdefault(member.id, [member]).append(partner)
        return list(groups.values())

    def _get_groups(self, system: SystemViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return the groups of one step."""
        agents = [agent for agent in system.agents(self._get_role('target')) if agent.alive]
        if self.groups == 'pairs':
            return self._get_partners(PartnerSpec('all', 'uniform', capacity=1), agents, system)
        if callable(self.groups) or isinstance(self.groups, PartnerSpec):
            return self._get_partners(self.groups, agents, system)
        if self.groups == 'partition':
            return self._get_partition(agents, self.size, system)
        if self.groups == 'neighborhood':
            return self._get_neighborhoods(agents)
        if self.groups == 'state':
            return self._get_shared(agents, system)
        raise self._get_variant_error(self.groups, 'groups', '_get_groups')

    def _act_on_groups(self, system: SystemViewProtocol) -> None:
        """Act on every group of one step in random order."""
        groups = self._get_groups(system)
        for index in system.rng.permutation(len(groups)):
            self.act_on(groups[int(index)], system)

    def _keyed(self, reduction: ReductionSpec, member: SimpleNamespace, members: SimpleNamespace) -> Expression:
        """Return a reduction over the members of a group that share the value of `by`.

        The reduction is restricted to these members.

        Args:
            reduction:
                The reduction over the members of the group.

            member:
                The symbols of the member.

            members:
                The symbols of the members of the group.

        Returns:
            reduction:
                The reduction restricted to the members that share the value of `by`.

        Raises:
            ValidationError:
                The reduction has weights.
        """
        if reduction.weights is not None:
            error_msg = f'The reduction {reduction.call} of the grouping {self.label!r} takes no weights.'
            raise ValidationError(error_msg)
        shared = SR(getattr(members, str(self.by))) == SR(getattr(member, str(self.by)))
        condition = shared if reduction.condition is None else both(reduction.condition, shared)
        if reduction.kind == 'count':
            return count(over=reduction.over, where=condition)
        return KEYED[reduction.kind](reduction.argument, over=reduction.over, where=condition)

    def _get_keyed_group(self, namespace: SimpleNamespace) -> tuple[SimpleNamespace, Expression]:
        """Return the namespace of the group by state of a member and its value.

        The reductions of the group are keyed.
        """
        target = self._get_member(namespace, 'target')
        names = {name: value for name, value in vars(namespace).items() if name != self._get_role('target')}
        group = SimpleNamespace(**names, member=target, members=target.other, others=())
        value = SR(self.function(group)) if self.function is not None else SR(0)
        keyed = {
            reduction.call: self._keyed(reduction, target, target.other)
            for reduction in reductions_in(value)
            if reduction.over is not None
        }
        return group, value.subs(keyed) if keyed else value

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the agent form, or the update of the state of group members.

        Args:
            namespace:
                The namespace of the symbols the laws can use, of one group in the symbolic form, and of the target for
                groups by state.

        Returns:
            laws:
                The rule that draws the groups and acts on them every step, or the update of `write`.
        """
        if self.form == 'agent':
            return [RuleLaw(self._act_on_groups, phase=self.phase)]
        if self.groups == 'state':
            group, value = self._get_keyed_group(namespace)
            return [UpdateLaw(getattr(group.member, str(self.write)), value, phase=self.phase)]
        value = self.function(namespace) if self.function is not None else 0
        return [UpdateLaw(getattr(namespace.member, str(self.write)), value, phase=self.phase)]

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Act on the members of one group.

        Args:
            members:
                The agent views of the members of the group.

            system:
                The system view of the rule.

        Raises:
            ValidationError:
                The grouping has neither an action nor its own `act_on`.
        """
        if self.act is None:
            error_msg = f'The grouping {self.label!r} has no action. Pass `act` or override `act_on`.'
            raise ValidationError(error_msg)
        self.act(members, system)
