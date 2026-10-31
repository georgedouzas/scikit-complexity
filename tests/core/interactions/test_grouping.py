"""Test the `_grouping` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import GroupingInteraction
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


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


def _voters(values: dict[str, object], size: int, **settings: object) -> GroupElement:
    """Return a group of voters."""
    return GroupElement('voters', element=OpinionElement, size=size, values=values, **settings)


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
