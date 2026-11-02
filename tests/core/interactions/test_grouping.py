"""Test the `_grouping` module of `skcomplex.core.interactions`."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import GroupingInteraction, ImitationInteraction
from skcomplex.core.laws.utils import count, mean, total, where
from skcomplex.core.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import BaseBackend, NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, PartnerSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


class OpinionElement(Element):
    """Member with an opinion and a count of the groups it joined."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec()] = 0.0,
        seen: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.seen = seen


class MajorityGroupingInteraction(GroupingInteraction):
    """Grouping whose groups adopt the opinion of their majority, ties to zero."""

    def act_on(self, members, model):
        """Set the opinion of every member to the majority of its group."""
        majority = float(2 * sum(member.s for member in members) > len(members))
        for member in members:
            member.s = majority


def _count(members, model):
    """Count the group for every member and record its size in the opinion."""
    for member in members:
        member.seen = member.seen + 1
        member.s = len(members)


def _run(system: System, until: int) -> Trajectory:
    """Return the trajectory of a system run to a time."""
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _voters(values: dict[str, object], size: int, location: object = None) -> GroupElement:
    """Return a group of voters."""
    return GroupElement('voters', element=OpinionElement, size=size, values=values, location=location)


def test_act_on_galam_groups_of_three_adopt_their_majority():
    """Test that in Galam groups of three every group ends with the opinion of its majority."""
    rng = np.random.default_rng(1)
    opinions = rng.integers(0, 2, 300).astype(float)
    snapshots = []

    def adopt(members, model):
        """Record the group, then set every member to the majority."""
        majority = float(2 * sum(member.s for member in members) > len(members))
        snapshots.append((len(members), [member.s for member in members], majority))
        for member in members:
            member.s = majority

    galam = GroupingInteraction('galam', target='voters', groups='partition', size=3, act=adopt)
    trajectory = _run(System(elements=[_voters({'s': opinions}, 300)], interactions=[galam]), 1)
    assert len(snapshots) == 100
    assert all(size == 3 and majority == float(sum(before) >= 2) for size, before, majority in snapshots)
    assert trajectory.agents('voters', 1).values['s'].sum() == sum(3 * majority for _, _, majority in snapshots)


def test_act_on_a_subclass_drives_groups_to_consensus():
    """Test that repeated majority groups of three drive a large majority to consensus."""
    opinions = np.r_[np.ones(70), np.zeros(30)]
    galam = MajorityGroupingInteraction('galam', target='voters', groups='partition', size=3)
    trajectory = _run(System(elements=[_voters({'s': opinions}, 100)], interactions=[galam]), 20)
    assert trajectory.agents('voters', 20).values['s'].tolist() == [1.0] * 100


def test_act_on_pairs_leaves_an_odd_member_out():
    """Test that pairs are disjoint and an odd member joins no pair."""
    pairing = GroupingInteraction('pairing', target='voters', groups='pairs', act=_count)
    values = _run(System(elements=[_voters({}, 5)], interactions=[pairing]), 1).agents('voters', 1).values
    assert sorted(values['seen'].tolist()) == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert set(values['s'][values['seen'] == 1].tolist()) == {2.0}


def test_act_on_the_remainder_of_a_partition_forms_a_smaller_group():
    """Test that a partition of seven into threes leaves a group of one."""
    partition = GroupingInteraction('partition', target='voters', groups='partition', size=3, act=_count)
    values = _run(System(elements=[_voters({}, 7)], interactions=[partition]), 1).agents('voters', 1).values
    assert sorted(values['s'].tolist()) == [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


def test_act_on_neighborhoods_groups_every_member_with_its_neighbors():
    """Test that every member of a full lattice forms one group with the members around it."""
    cells = [(row, column) for row in range(3) for column in range(3)]
    voters = _voters({}, 9, location=cells)
    grouping = GroupingInteraction('local', target='voters', groups='neighborhood', act=_count)
    space = LatticeSpace((3, 3), neighborhood='von_neumann')
    values = _run(System(elements=[voters], interactions=[grouping], space=space), 1).agents('voters', 1).values
    assert values['seen'].tolist() == [5.0] * 9


def test_act_on_groups_by_state_take_at_most_the_capacity():
    """Test that members sharing a value form one group cut to the capacity."""
    sides = {'s': [0.0, 0.0, 0.0, 1.0, 1.0]}
    grouping = GroupingInteraction('sides', target='voters', groups='state', by='s', capacity=2, act=_count)
    values = _run(System(elements=[_voters(sides, 5)], interactions=[grouping]), 1).agents('voters', 1).values
    assert sorted(values['seen'].tolist()) == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_act_on_without_an_action_raises():
    """Test that a grouping without an action or its own method raises an error."""
    with pytest.raises(ValidationError, match='no action'):
        GroupingInteraction('groups', target='voters').act_on([], None)


@pytest.mark.parametrize(
    'arguments',
    [{'groups': 'clusters'}, {'size': 1}, {'size': True}, {'groups': 'state'}, {'capacity': 0}, {'act': 3}],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that unknown groups, a small size, a missing state, a capacity below one, and a bad action are rejected."""
    with pytest.raises(ValidationError):
        GroupingInteraction('groups', target='voters', **arguments).check()


class CouncilInteraction(GroupingInteraction):
    """Grouping that binds its members as `council`."""

    def __init__(self, label: str, *, council: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=council, **arguments)
        self.council = council

    def _get_role(self, role: str) -> str:
        """Return `council` for the target."""
        return 'council' if role == 'target' else role


def _symbolic(
    groups: str,
    function: Callable[[SimpleNamespace], object],
    *,
    size: int = 2,
    by: str | None = None,
    reduce: str = 'sum',
) -> GroupingInteraction:
    """Return a symbolic grouping of the voters that writes their opinion."""
    return GroupingInteraction(
        'groups',
        target='voters',
        groups=groups,
        size=size,
        by=by,
        form='symbolic',
        write='s',
        function=function,
        reduce=reduce,
    )


class PlayerElement(Element):
    """Player with a strategy, a payoff, and a small bias of its payoff that breaks ties."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        state: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
        bias: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.state = state
        self.payoff = payoff
        self.bias = bias


def _public_good(group: SimpleNamespace) -> object:
    """Return the share of the public good of a group of five minus the contribution, with a fifth of the bias."""
    return 3.8 * total(group.members.state, over=group.members) / 5 - group.member.state + group.member.bias / 5


def _majority(group: SimpleNamespace) -> object:
    """Return 1 when more than half of the members of a group hold 1, and 0 otherwise."""
    return where(mean(group.members.s, over=group.members) > 1 / 2, 1, 0)


def _total(group: SimpleNamespace) -> object:
    """Return the total opinion of the members of a group."""
    return total(group.members.s, over=group.members)


def _size(group: SimpleNamespace) -> object:
    """Return the number of members of a group."""
    return count(over=group.members)


def _weighted(group: SimpleNamespace) -> object:
    """Return the opinion of the member of a group times the total opinion of the group."""
    return group.member.s * total(group.members.s, over=group.members)


def _stream(seed: int = 0) -> np.random.Generator:
    """Return the generator a simulation with a seed hands its solver."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def _symbolic_run(system: System, until: int, backend: BaseBackend, seed: int = 0) -> np.ndarray:
    """Return the opinions of the voters at every frame of a run of a symbolic grouping."""
    simulation = Simulation(system, random_state=seed, backend=backend).check().prepare().run(until=until)
    return simulation.trajectory_.values('voters.s')


@BACKENDS
def test_get_laws_symbolic_partitions_set_every_group_to_its_majority(backend):
    """Test that after one step every group of the drawn partition holds the majority opinion it had."""
    opinions = np.random.default_rng(1).integers(0, 2, 30).astype(float)
    galam = _symbolic('partition', _majority, size=3)
    values = _symbolic_run(System(elements=[_voters({'s': opinions}, 30)], interactions=[galam]), 1, backend)
    groups = _stream().permutation(30).reshape(10, 3)
    expected = np.repeat((opinions[groups].sum(axis=1) >= 2).astype(float), 3)
    assert values[1][groups.ravel()].tolist() == expected.tolist()


@BACKENDS
def test_get_laws_symbolic_partitions_are_drawn_from_the_random_stream(backend):
    """Test that two runs with one seed draw the same partitions, and runs with two seeds draw different ones."""
    opinions = np.random.default_rng(2).integers(0, 2, 60).astype(float)
    system = System(elements=[_voters({'s': opinions}, 60)], interactions=[_symbolic('partition', _majority, size=3)])
    first, second, third = (_symbolic_run(system, 3, backend, seed) for seed in (0, 0, 1))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


@BACKENDS
def test_get_laws_symbolic_pairs_leave_an_odd_member_unchanged(backend):
    """Test that the members of a pair take the total of the pair and the odd member out keeps its value."""
    voters = _voters({'s': [1.0, 2.0, 3.0, 4.0, 5.0]}, 5)
    values = _symbolic_run(System(elements=[voters], interactions=[_symbolic('pairs', _total)]), 1, backend)
    order = _stream().permutation(5)
    expected = np.arange(1.0, 6.0)
    for first, second in order[:4].reshape(2, 2):
        expected[[first, second]] = first + second + 2
    assert values[1].tolist() == expected.tolist()


@BACKENDS
def test_get_laws_symbolic_remainder_of_a_partition_forms_a_smaller_group(backend):
    """Test that a partition of seven into threes gives every member the size of its group."""
    values = _symbolic_run(System(elements=[_voters({}, 7)], interactions=[_symbolic('partition', _size)]), 1, backend)
    assert sorted(values[1].tolist()) == [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]


@BACKENDS
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_get_laws_symbolic_neighborhoods_reduce_over_the_groups_of_a_member(backend, reduce):
    """Test that a member takes the sum or the mean over its neighbourhood groups of a function of each group."""
    values = np.arange(16.0)
    space = LatticeSpace((4, 4), neighborhood='von_neumann', boundary='fixed')
    grid = LatticeElement('voters', element=OpinionElement, space=space, values={'s': values})
    grouping = _symbolic('neighborhood', _weighted, reduce=reduce)
    result = _symbolic_run(System(elements=[grid], interactions=[grouping]), 1, backend)[1]
    rows, columns = np.divmod(np.arange(16), 4)
    groups = [
        [site, *np.flatnonzero(np.abs(rows - rows[site]) + np.abs(columns - columns[site]) == 1).tolist()]
        for site in range(16)
    ]
    contributions = [[values[site] * values[group].sum() for group in groups if site in group] for site in range(16)]
    expected = [np.sum(items) if reduce == 'sum' else np.mean(items) for items in contributions]
    assert np.allclose(result, expected, atol=1e-12)


@BACKENDS
def test_get_laws_symbolic_groups_by_state_reduce_over_the_members_sharing_it(backend):
    """Test that groups by state give every member the total of the members that share its value of the state."""
    keyed = _symbolic('state', _total, by='seen')
    voters = _voters({'s': [1.0, 2.0, 3.0, 4.0, 5.0], 'seen': [0.0, 1.0, 0.0, 1.0, 1.0]}, 5)
    values = _symbolic_run(System(elements=[voters], interactions=[keyed]), 1, backend)
    assert values[1].tolist() == [4.0, 11.0, 4.0, 11.0, 11.0]


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'act': _count, 'write': 's', 'function': _majority}, '`act`, a setting of the agent form'),
        ({'capacity': 2, 'write': 's', 'function': _majority}, '`capacity`, a setting of the agent form'),
        ({'write': 's'}, 'Pass the name of the state'),
        ({'write': 's', 'function': _majority, 'reduce': 'max'}, 'Unknown reduction'),
    ],
)
def test_check_rejects_settings_of_the_symbolic_form(arguments, match):
    """Test that a symbolic grouping with an agent setting, no function, or an unknown reduction raises an error."""
    with pytest.raises(ValidationError, match=match):
        GroupingInteraction('groups', target='voters', form='symbolic', **arguments).check()


@pytest.mark.parametrize('arguments', [{'form': 'mixed'}, {'write': 's'}, {'function': _majority}])
def test_check_rejects_an_unknown_form_and_symbolic_settings_of_the_agent_form(arguments):
    """Test that an unknown form, and a state or function given to the agent form, raise an error."""
    with pytest.raises(ValidationError):
        GroupingInteraction('groups', target='voters', **arguments).check()


@BACKENDS
def test_get_laws_symbolic_phases_imitate_the_payoffs_of_the_same_step(backend):
    """Test that a payoff phase and an imitation phase in one step equal a loop of the payoffs and then the copies."""
    rng = np.random.default_rng(3)
    strategies, bias = rng.integers(0, 2, 36).astype(float), rng.uniform(0, 0.01, 36)
    space = LatticeSpace((6, 6), neighborhood='von_neumann', boundary='periodic')
    players = LatticeElement('players', element=PlayerElement, space=space, values={'state': strategies, 'bias': bias})
    payoffs = GroupingInteraction(
        'goods',
        target='players',
        groups='neighborhood',
        form='symbolic',
        write='payoff',
        function=_public_good,
        phase='play',
    )
    imitation = ImitationInteraction('imitation', target='players', rule='best', form='probability', phase='imitate')
    time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
    system = System(elements=[players], interactions=[payoffs, imitation], time=time)
    simulation = Simulation(system, random_state=0, backend=backend).check().prepare().run(until=3)
    around = [
        np.array(
            [
                ((row + 1) % 6) * 6 + column,
                ((row - 1) % 6) * 6 + column,
                row * 6 + (column + 1) % 6,
                row * 6 + (column - 1) % 6,
            ],
        )
        for row, column in zip(*np.divmod(np.arange(36), 6), strict=True)
    ]
    state = strategies.copy()
    for step in range(1, 4):
        goods = np.array([3.8 * (state[site] + state[around[site]].sum()) / 5 for site in range(36)])
        payoff = np.array([goods[site] + goods[around[site]].sum() - 5 * state[site] for site in range(36)]) + bias
        best = [around[site][np.argmax(payoff[around[site]])] for site in range(36)]
        state = np.array(
            [state[best[site]] if payoff[best[site]] > payoff[site] else state[site] for site in range(36)],
        )
        assert np.allclose(simulation.trajectory_.values('players.payoff')[step], payoff, atol=1e-12)
        assert simulation.trajectory_.values('players.state')[step].tolist() == state.tolist()


@pytest.mark.parametrize('groups', ['partition', 'neighborhood'])
def test_get_laws_symbolic_reads_a_renamed_target(groups):
    """Test that a symbolic grouping that binds its members as `council` writes the values of the plain grouping."""
    space = LatticeSpace((4, 4), neighborhood='von_neumann', boundary='fixed')
    grid = LatticeElement('voters', element=OpinionElement, space=space, values={'s': np.arange(16.0) % 3})
    arguments = {'groups': groups, 'size': 2, 'form': 'symbolic', 'write': 's', 'function': _weighted}
    plain = GroupingInteraction('groups', target='voters', **arguments)
    renamed = CouncilInteraction('groups', council='voters', **arguments)
    values = [
        _symbolic_run(System(elements=[grid], interactions=[item]), 2, NumpyBackend()) for item in (plain, renamed)
    ]
    np.testing.assert_allclose(values[1], values[0])


def _groups_after_one_step(groups: object, seed: int) -> list[list[float]]:
    """Return the values of the members of every group of one step, in the order they act."""
    seen: list[list[float]] = []

    def record(members, model):
        """Record the members of a group."""
        seen.append([member.s for member in members])

    voters = _voters({'s': [float(index) for index in range(7)]}, 7)
    grouping = GroupingInteraction('groups', target='voters', groups=groups, act=record)
    system = System(elements=[voters], interactions=[grouping], time=DiscreteTime())
    Simulation(system, random_state=seed).check().prepare().run(until=1)
    return seen


def test_get_groups_pairs_is_a_shorthand_of_a_partner_spec():
    """Test that the pairs and their partner spec draw identical groups over 20 seeds."""
    for seed in range(20):
        assert _groups_after_one_step('pairs', seed) == _groups_after_one_step(
            PartnerSpec('all', 'uniform', capacity=1),
            seed,
        )


def test_get_groups_collects_every_member_with_its_partners():
    """Test that a partner spec that takes every candidate groups a member with all the others."""
    groups = _groups_after_one_step(PartnerSpec('all', 'all'), 0)
    assert len(groups) == 7
    assert {len(group) for group in groups} == {7}


def _first_two(left, right, rng, capacity):
    """Return one pair of the first two members."""
    del right, rng, capacity
    return [(left[0], left[1])]


def test_get_groups_takes_a_pairing_function():
    """Test that a pairing function gives the groups of the step."""
    assert _groups_after_one_step(_first_two, 0) == [[0.0, 1.0]]
