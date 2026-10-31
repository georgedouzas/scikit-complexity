"""Define groupings, which draw groups of the agents of a group every step and act on each group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import RoleSpec
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

GROUPS = ('pairs', 'partition', 'neighborhood', 'state')
PAIR = 2

type Act = Callable[[list[AgentViewProtocol], ModelViewProtocol], None]


class GroupingInteraction(Interaction):
    """Grouping of the agents of a group into groups drawn every step, with an action on every group.

    The groups are one of four kinds:

    - `'pairs'` draws disjoint random pairs, and an odd agent out forms no group.
    - `'partition'` draws a random partition into groups of `size` agents, the last group holding the remainder.
    - `'neighborhood'` forms one group per agent, the agent first and then the agents in the cells or nodes around it.
    - `'state'` forms one group per value of the state `by`, of at most `capacity` agents drawn at random among those
      that hold the value.

    The groups act in random order, each through `act(members, model)`, a function of the list of agent views and the
    model view, or the method `act_on` a subclass overrides. The grouping is an agent rule, so the system runs in
    discrete time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group whose agents are grouped.

        groups:
            The kind of groups, `'pairs'`, `'partition'`, `'neighborhood'`, or `'state'`.

        act:
            The function of the members of a group and the model view that acts on the group, used when a subclass
            does not override `act_on`.

        size:
            The number of agents of a group of a partition.

        by:
            The name of the state whose values the groups `'state'` share.

        capacity:
            The largest number of agents of a group `'state'`, or `None` for no limit.

        phase:
            The phase of the step the grouping runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The kind of groups is unknown, the size is below two, the groups `'state'` have no state, or the capacity
            is not positive, raised by `check`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import GroupingInteraction
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> class VoterElement(Element):
        ...     def __init__(self, label, visual=None, *, s: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label, visual=visual)
        ...         self.s = s
        >>> def adopt_majority(members, model):
        ...     majority = float(2 * sum(member.s for member in members) > len(members))
        ...     for member in members:
        ...         member.s = majority
        >>> voters = GroupElement('voters', element=VoterElement, size=3, values={'s': [1.0, 1.0, 0.0]})
        >>> galam = GroupingInteraction('galam', target='voters', groups='partition', size=3, act=adopt_majority)
        >>> trajectory = Simulation(System(elements=[voters], interactions=[galam])).check().prepare().run(until=1)
        >>> trajectory.trajectory_.agents('voters', 1).values['s'].tolist()
        [1.0, 1.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        groups: str = 'pairs',
        act: Act | None = None,
        size: int = 2,
        by: str | None = None,
        capacity: int | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.groups = groups
        self.act = act
        self.size = size
        self.by = by
        self.capacity = capacity
        self.phase = phase

    def _check(self) -> None:
        """Raise an error for an unknown kind of groups, a size below two, a missing state, or a capacity below one.

        Raises:
            ValidationError:
                The kind of groups is unknown, the size is below two, the groups `'state'` have no state, or the
                capacity is not positive.
        """
        super()._check()
        if self.groups not in GROUPS:
            error_msg = f'Unknown groups {self.groups!r} of {self.label!r}. Use one of {", ".join(GROUPS)}.'
            raise ValidationError(error_msg)
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

    def _get_partition(
        self,
        agents: Sequence[AgentViewProtocol],
        size: int,
        model: ModelViewProtocol,
    ) -> list[list[AgentViewProtocol]]:
        """Return a random partition of agents into groups of a size, with the remainder in the last group."""
        order = [agents[int(index)] for index in model.rng.permutation(len(agents))]
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
        model: ModelViewProtocol,
    ) -> list[list[AgentViewProtocol]]:
        """Return the groups of agents that share a value of the state `by`, each cut at random to the capacity."""
        shared: dict[float, list[AgentViewProtocol]] = {}
        for agent in agents:
            shared.setdefault(float(getattr(agent, str(self.by))), []).append(agent)
        groups = []
        for value in sorted(shared):
            members = shared[value]
            order = [members[int(index)] for index in model.rng.permutation(len(members))]
            groups.append(order if self.capacity is None else order[: self.capacity])
        return groups

    def _get_groups(self, model: ModelViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return the groups of one step."""
        agents = [agent for agent in model.agents('target') if agent.alive]
        if self.groups == 'pairs':
            return [group for group in self._get_partition(agents, PAIR, model) if len(group) == PAIR]
        if self.groups == 'partition':
            return self._get_partition(agents, self.size, model)
        if self.groups == 'neighborhood':
            return self._get_neighborhoods(agents)
        return self._get_shared(agents, model)

    def _act_on_groups(self, model: ModelViewProtocol) -> None:
        """Act on every group of one step in random order."""
        groups = self._get_groups(model)
        for index in model.rng.permutation(len(groups)):
            self.act_on(groups[int(index)], model)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the grouping.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule that draws the groups and acts on them every step.
        """
        return [RuleLaw(self._act_on_groups, phase=self.phase)]

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Act on the members of one group.

        Args:
            members:
                The agent views of the members of the group.

            model:
                The model view of the rule.

        Raises:
            ValidationError:
                The grouping has neither an action nor its own `act_on`.
        """
        if self.act is None:
            error_msg = f'The grouping {self.label!r} has no action. Pass `act` or override `act_on`.'
            raise ValidationError(error_msg)
        self.act(members, model)
