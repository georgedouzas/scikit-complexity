"""Test the `_birth_death` module of `skcomplex.systems.general.agents`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws import Sink, Source
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from skcomplex.systems.general.agents import BirthDeathInteraction


class ResourceElement(Element):
    """Cell that holds a resource."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        resource: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.resource = resource


class ReserveElement(Element):
    """Member that holds a reserve and a trait."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        reserve: Annotated[float, StateSpec()] = 0.0,
        trait: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.reserve = reserve
        self.trait = trait


def _feeding(uptake: float, gain: float) -> TransferInteraction:
    """Return the uptake of the resource of a cell by the members in it, of which they gain a share."""
    return TransferInteraction(
        'feeding',
        source='cells',
        target='foragers',
        state='resource',
        into=Sink(),
        amount=lambda cell, forager, system: min(cell.resource, uptake),
        legs=((Source(), 'reserve', lambda cell, forager, system: gain * min(cell.resource, uptake)),),
        pairing='cell',
        form='agent',
    )


def _upkeep(amount: float) -> TransferInteraction:
    """Return the upkeep every forager pays from its reserve every step."""
    return TransferInteraction(
        'upkeep',
        source='foragers',
        target='foragers',
        state='reserve',
        into=Sink(),
        amount=amount,
        form='agent',
    )


def _life(threshold: float = 0.0, probability: float = 0.0) -> BirthDeathInteraction:
    """Return the death of foragers below zero reserve and their split above a threshold with a probability."""
    return BirthDeathInteraction(
        'life',
        target='foragers',
        death=lambda forager, system: forager.reserve < 0,
        birth=lambda forager, system: probability if forager.reserve >= threshold else 0.0,
        split=('reserve',),
    )


def _forage(
    until: int,
    interactions: list,
    resource: float,
    reserve: float,
    size: int = 1,
) -> Trajectory:
    """Return the trajectory of foragers in the one cell of a lattice that holds a resource."""
    cells = GroupElement('cells', element=ResourceElement, size=1, values={'resource': resource}, location=[(0, 0)])
    foragers = GroupElement(
        'foragers',
        element=ReserveElement,
        size=size,
        values={'reserve': reserve},
        location=[(0, 0)] * size,
        variable=True,
    )
    system = CompositeSystem(elements=[cells, foragers], interactions=interactions, space=LatticeSpace((1, 1)))
    return simulate(system, until=until, random_state=0)


def _reserves(trajectory: Trajectory, frame: int) -> list[float]:
    """Return the reserves of the foragers alive at a frame."""
    return trajectory.agents('foragers', frame).values['reserve'].tolist()


def test_live_forages_as_a_composition_of_transfers_and_a_life_cycle():
    """Test that a forager takes at most its uptake from its cell and gains it times the gain."""
    trajectory = _forage(2, [_feeding(uptake=2.0, gain=0.5), _life()], resource=3.0, reserve=0.0)
    assert trajectory.agents('cells', 2).values['resource'].tolist() == [0.0]
    assert _reserves(trajectory, 1) == [1.0]
    assert _reserves(trajectory, 2) == [1.5]


def test_live_shares_a_scarce_resource_between_foragers():
    """Test that two foragers of one cell take no more than the cell holds."""
    trajectory = _forage(1, [_feeding(uptake=1.0, gain=1.0)], resource=1.5, reserve=0.0, size=2)
    assert sorted(_reserves(trajectory, 1)) == [0.5, 1.0]


def test_live_removes_foragers_below_zero_reserve():
    """Test that a forager survives at zero reserve and dies when its upkeep takes its reserve below zero."""
    trajectory = _forage(2, [_upkeep(1.0), _life()], resource=0.0, reserve=1.0)
    assert _reserves(trajectory, 1) == [0.0]
    assert _reserves(trajectory, 2) == []


def test_live_member_schedule_runs_the_upkeep_and_the_death_of_each_forager_together():
    """Test that a forager pays its upkeep and dies below zero in its own activation, under the member schedule."""
    upkeep = TransferInteraction(
        'upkeep',
        source='foragers',
        target='foragers',
        state='reserve',
        into=Sink(),
        amount=1.0,
        schedule='member',
        form='agent',
    )
    life = BirthDeathInteraction(
        'life',
        target='foragers',
        death=lambda forager, system: forager.reserve < 0,
        schedule='member',
    )
    trajectory = _forage(2, [upkeep, life], resource=0.0, reserve=1.0, size=3)
    assert _reserves(trajectory, 1) == [0.0] * 3
    assert _reserves(trajectory, 2) == []


def test_live_splits_the_reserve_above_the_threshold():
    """Test that a forager at its threshold splits its reserve with an offspring, and one below it does not."""
    split = _forage(1, [_upkeep(1.0), _life(threshold=6.0, probability=1.0)], resource=0.0, reserve=8.0)
    kept = _forage(1, [_upkeep(1.0), _life(threshold=6.0, probability=1.0)], resource=0.0, reserve=6.0)
    assert _reserves(split, 1) == [3.5, 3.5]
    assert _reserves(kept, 1) == [5.0]


def test_live_feeds_foragers_on_the_nodes_of_a_network():
    """Test that foragers on a network take the resource of their own node."""
    cells = GroupElement('cells', element=ResourceElement, size=2, values={'resource': [1.0, 4.0]}, location=[0, 1])
    foragers = GroupElement('foragers', element=ReserveElement, size=1, location=[1])
    system = CompositeSystem(
        elements=[cells, foragers],
        interactions=[_feeding(uptake=10.0, gain=1.0)],
        space=NetworkSpace(nx.path_graph(2)),
    )
    trajectory = simulate(system, until=1, random_state=0)
    assert _reserves(trajectory, 1) == [4.0]
    assert trajectory.agents('cells', 1).values['resource'].tolist() == [1.0, 0.0]


def test_live_rate_draws_a_poisson_number_of_births():
    """Test that births at a rate follow a Poisson number of that mean, each with the offspring count."""
    parents = GroupElement('parents', element=ReserveElement, size=400, variable=True, capacity=4000)
    births = BirthDeathInteraction('births', target='parents', rate=1.5, offspring=2)
    system = CompositeSystem(elements=[parents], interactions=[births])
    trajectory = simulate(system, until=1, random_state=0)
    assert len(trajectory.agents('parents', 1).values['reserve']) / 400 - 1 == pytest.approx(3.0, rel=0.1)


def test_live_places_offspring_in_free_neighboring_cells():
    """Test that offspring placed in neighbouring cells take free cells only, and fewer are born when none is free."""
    parents = GroupElement('parents', element=ReserveElement, size=1, location=[(1, 1)], variable=True)
    births = BirthDeathInteraction('births', target='parents', birth=1.0, offspring=12, placement='neighbor')
    space = LatticeSpace((3, 3), metric='chebyshev')
    trajectory = simulate(CompositeSystem(elements=[parents], interactions=[births], space=space), until=1)
    locations = {tuple(cell) for cell in np.asarray(trajectory.agents('parents', 1).locations).tolist()}
    assert len(locations) == 9


def test_live_inherits_and_mutates_the_states_of_offspring():
    """Test that offspring copy the inherited states of their parent, changed by the mutation."""

    def mutate(values, system):
        """Raise the inherited trait by one."""
        return {**values, 'trait': values['trait'] + 1.0}

    parents = GroupElement(
        'parents',
        element=ReserveElement,
        size=1,
        values={'trait': 2.0, 'reserve': 5.0},
        variable=True,
    )
    births = BirthDeathInteraction('births', target='parents', birth=1.0, inherit=('trait',), mutate=mutate)
    trajectory = simulate(CompositeSystem(elements=[parents], interactions=[births]), until=1)
    values = trajectory.agents('parents', 1).values
    assert values['trait'].tolist() == [2.0, 3.0]
    assert values['reserve'].tolist() == [5.0, 0.0]


def _cycle(life: BirthDeathInteraction, *groups: GroupElement, until: int = 1) -> Trajectory:
    """Return the trajectory of groups of reserve members under a life cycle on a lattice."""
    system = CompositeSystem(elements=list(groups), interactions=[life], space=LatticeSpace((3, 3)))
    return simulate(system, until=until, random_state=0)


def test_live_creates_offspring_in_another_group_with_given_values():
    """Test that births into another group create members of it with the values of the parent's function."""
    parents = GroupElement('parents', element=ReserveElement, size=2, values={'trait': [1.0, 2.0]})
    young = GroupElement('young', element=ReserveElement, size=1, location=[(0, 0)], variable=True)
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=1.0,
        group='young',
        values=lambda parent, system: {'trait': parent.trait * 10},
    )
    trajectory = _cycle(life, parents, young)
    assert sorted(trajectory.agents('young', 1).values['trait'].tolist()) == [0.0, 10.0, 20.0]
    assert len(trajectory.agents('parents', 1).values['trait']) == 2
    assert trajectory.values('life.births')[-1] == 2


def test_live_replaces_parents_by_their_offspring():
    """Test that with replacement a parent is removed after its births, also when it has no offspring."""
    parents = GroupElement('parents', element=ReserveElement, size=4, variable=True)
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=1.0,
        offspring=lambda parent, system: 2 if parent.id % 2 else 0,
        values=lambda parent, system: {'trait': parent.trait + 1},
        replace=True,
    )
    values = _cycle(life, parents).agents('parents', 1).values
    assert values['trait'].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_live_keeps_a_share_of_the_split_states():
    """Test that a parent keeps the share `keep` of a split state and its offspring take the rest."""
    parents = GroupElement('parents', element=ReserveElement, size=1, values={'reserve': 8.0}, variable=True)
    life = BirthDeathInteraction('life', target='parents', birth=1.0, offspring=2, split=('reserve',), keep=0.5)
    assert _reserves_of(_cycle(life, parents)) == [2.0, 2.0, 4.0]


def test_live_resets_a_parent_after_its_birth():
    """Test that a parent takes the reset values after it gives birth, and a parent that does not keeps its own."""
    parents = GroupElement('parents', element=ReserveElement, size=2, values={'reserve': [3.0, 1.0]}, variable=True)
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=lambda parent, system: parent.reserve >= 2,
        reset={'reserve': 0.0},
        values=lambda parent, system: {'reserve': 0.5},
    )
    assert _reserves_of(_cycle(life, parents)) == [0.0, 0.5, 1.0]


def test_live_renews_dying_agents_in_place():
    """Test that a renewed agent keeps its identity and the states the new values leave out."""
    parents = GroupElement(
        'parents',
        element=ReserveElement,
        size=2,
        values={'reserve': [-1.0, 2.0], 'trait': [7.0, 8.0]},
        variable=True,
    )
    life = BirthDeathInteraction(
        'life',
        target='parents',
        death=lambda agent, system: agent.reserve < 0,
        renew=True,
        values=lambda agent, system: {'reserve': 1.0},
    )
    trajectory = _cycle(life, parents)
    values = trajectory.agents('parents', 1).values
    assert values['reserve'].tolist() == [1.0, 2.0]
    assert values['trait'].tolist() == [7.0, 8.0]
    assert trajectory.values('life.deaths')[-1] == 1


def test_live_culls_kin_of_another_group():
    """Test that an agent removes the first member of the other group that is its kin."""
    owners = GroupElement('owners', element=ReserveElement, size=2, values={'trait': [0.0, 1.0]})
    owned = GroupElement('owned', element=ReserveElement, size=3, values={'trait': [1.0, 0.0, 1.0]}, variable=True)
    life = BirthDeathInteraction(
        'life',
        target='owners',
        group='owned',
        cull=lambda owner, system: owner.trait == 1.0,
        kin=lambda owner, member: member.trait == owner.trait,
    )
    assert _cycle(life, owners, owned).agents('owned', 1).values['trait'].tolist() == [0.0, 1.0]


def test_live_brings_arrivals_at_placed_locations():
    """Test that a Poisson number of arrivals join the group at the placement's locations with drawn values."""
    members = GroupElement('members', element=ReserveElement, size=1, location=[(2, 1)], variable=True)
    life = BirthDeathInteraction(
        'life',
        target='members',
        arrival=lambda system: 50.0,
        placement=lambda parent, system: (2, 1),
        values=lambda parent, system: {'trait': float(system.rng.integers(5))},
    )
    trajectory = _cycle(life, members, until=4)
    snapshot = trajectory.agents('members', 4)
    assert (len(snapshot.values['trait']) - 1) / 4 == pytest.approx(50.0, rel=0.2)
    assert {tuple(cell) for cell in np.asarray(snapshot.locations).tolist()} == {(2, 1)}
    assert set(snapshot.values['trait'].tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}


def _reserves_of(trajectory: Trajectory) -> list[float]:
    """Return the sorted reserves of the parents at the first frame."""
    return sorted(trajectory.agents('parents', 1).values['reserve'].tolist())


@pytest.mark.parametrize(
    'arguments',
    [
        {'placement': 'far'},
        {'birth': 0.5, 'rate': 1.0},
        {'death': 'old'},
        {'mutate': 3},
        {'keep': 'half'},
        {'values': 1.0},
        {'cull': 'half'},
        {'pick': 'last'},
        {'schedule': 'parallel'},
        {'schedule': 'member', 'arrival': 1.0},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown placement, two kinds of birth, and values that are not numbers or functions are rejected."""
    with pytest.raises(ValidationError):
        BirthDeathInteraction('life', target='foragers', **arguments).prepare()


def test_live_culls_a_random_member_of_its_group_after_a_replacement():
    """Test that a parent replaced by its offspring still removes a random member of its own group."""
    parents = GroupElement('parents', element=ReserveElement, size=4, values={'trait': 1.0}, variable=True)
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=1.0,
        offspring=2,
        inherit=('trait',),
        replace=True,
        cull=1.0,
        pick='random',
        schedule='member',
    )
    system = CompositeSystem(
        elements=[parents],
        interactions=[life],
        space=LatticeSpace((3, 3)),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('default', activation='visits', visits=0.5)])),
    )
    trajectory = simulate(system, until=1, random_state=0)
    assert len(trajectory.agents('parents', 1).ids) == 4


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'placement': PartnerSpec()}, 'Unknown placement .*a function'),
        ({'pick': PartnerSpec()}, "Unknown pick .*'first', 'random'"),
    ],
)
def test_variant_refuses_a_spec_and_lists_what_it_takes(arguments, message):
    """Test that a placement and a pick refuse a spec, and list their names and functions."""
    with pytest.raises(ValidationError, match=message):
        BirthDeathInteraction('a', target='b', **arguments).prepare()
