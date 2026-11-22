"""Test the `_birth_death` module of `skcomplex.systems.general.agents`."""

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import PartnerSpec, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from skcomplex.core.components.types import AgentViewProtocol, SystemViewProtocol
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.general.agents import BirthDeathInteraction

LATTICE = LatticeSpace((3, 3))


@runtime_checkable
class CellView(AgentViewProtocol, Protocol):
    """View of a cell, with its resource."""

    resource: float


class ForagerView(AgentViewProtocol, Protocol):
    """View of a forager, with its reserve."""

    reserve: float


class ReserveElement(Element):
    """Members that hold a reserve and a trait, with places given as values named after the axes of their space."""

    def __init__(
        self,
        label: str,
        *,
        reserve: object = 0.0,
        trait: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
        values: dict[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.reserve = reserve
        self.trait = trait


def _places(space: BaseSpace, places: object) -> dict[str, object]:
    """Return the values named after the axes of a space that place members.

    A list gives one place per member, and a single cell or node places every member there.
    """
    axes = space.prepare().axes_
    if isinstance(places, list):
        points = [place if isinstance(place, tuple) else (place,) for place in places]
        return {axis: [point[index] for point in points] for index, axis in enumerate(axes)}
    point = places if isinstance(places, tuple) else (places,)
    return dict(zip(axes, point, strict=True))


def _members(
    label: str,
    size: int,
    *,
    space: BaseSpace | None = None,
    places: object = None,
    **values: object,
) -> ReserveElement:
    """Return members with a reserve and a trait of their own, zero by default, at places of a space if given."""
    own = {name: np.broadcast_to(np.asarray(value, dtype=float), size).copy() for name, value in values.items()}
    located = None if space is None or places is None else _places(space, places)
    return ReserveElement(label, **own, size=size, space=space, values=located)


def _resources(space: BaseSpace, places: object, resource: object) -> Element:
    """Return cells that hold a resource at places of a space, one cell for a single place."""
    size = len(places) if isinstance(places, list) else 1
    return Element('cells', values={'resource': resource, **_places(space, places)}, size=size, space=space)


def _feeding(uptake: float, gain: float, regrowth: float = 0.0) -> Interaction:
    """Return the uptake of the resource of a cell by the members in it, of which they gain a share.

    The resource of every cell grows back by `regrowth` each step.
    """

    def regrow(cell: CellView, system: SystemViewProtocol) -> None:
        cell.resource += regrowth

    def feed(forager: ForagerView, system: SystemViewProtocol) -> None:
        groups = set(system.get_groups('cells'))
        for cell in [other for other in forager.get_agents_here() if other.group in groups]:
            assert isinstance(cell, CellView)
            if forager.alive and cell.alive:
                taken = min(cell.resource, uptake)
                cell.resource -= taken
                forager.reserve += gain * taken

    return Interaction(
        'feeding',
        roles={'target': 'foragers', 'cells': 'cells'},
        laws=lambda s: [UpdateLaw(s.cells, regrow), UpdateLaw(s.target, feed)],
    )


def _upkeep(amount: float) -> Interaction:
    """Return the upkeep every forager pays from its reserve every step."""

    def pay(forager: ForagerView, system: SystemViewProtocol) -> None:
        forager.reserve -= amount

    return Interaction('upkeep', roles={'target': 'foragers'}, laws=lambda s: [UpdateLaw(s.target, pay)])


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
    space = LatticeSpace((1, 1))
    cells = _resources(space, (0, 0), resource)
    foragers = _members('foragers', size, space=space, places=(0, 0), reserve=reserve)
    system = System(elements=[cells, foragers], interactions=interactions)
    return system.prepare().simulate(until, random_state=0)


def _reserves(trajectory: Trajectory, frame: int) -> list[float]:
    """Return the reserves of the foragers alive at a frame."""
    return trajectory.get_agents('foragers', frame).values['reserve'].tolist()


def test_live_forages_as_a_composition_of_transfers_and_a_life_cycle():
    """Test that a forager takes at most its uptake from its cell and gains it times the gain."""
    trajectory = _forage(2, [_feeding(uptake=2.0, gain=0.5), _life()], resource=3.0, reserve=0.0)
    assert trajectory.get_agents('cells', 2).values['resource'].tolist() == [0.0]
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
    upkeep = _upkeep(1.0)
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
    space = NetworkSpace(nx.path_graph(2))
    cells = _resources(space, [0, 1], [1.0, 4.0])
    foragers = _members('foragers', 1, space=space, places=1)
    system = System(elements=[cells, foragers], interactions=[_feeding(uptake=10.0, gain=1.0)])
    trajectory = system.prepare().simulate(1, random_state=0)
    assert _reserves(trajectory, 1) == [4.0]
    assert trajectory.get_agents('cells', 1).values['resource'].tolist() == [1.0, 0.0]


def test_live_rate_draws_a_poisson_number_of_births():
    """Test that births at a rate follow a Poisson number of that mean, each with the offspring count."""
    parents = _members('parents', 400)
    births = BirthDeathInteraction('births', target='parents', rate=1.5, offspring=2)
    trajectory = System(elements=[parents], interactions=[births]).prepare().simulate(1, random_state=0)
    assert len(trajectory.get_agents('parents', 1).values['reserve']) / 400 - 1 == pytest.approx(3.0, rel=0.1)


def test_live_places_offspring_in_free_neighboring_cells():
    """Test that offspring placed in neighbouring cells take free cells only, and fewer are born when none is free."""
    parents = _members('parents', 1, space=LatticeSpace((3, 3), metric='chebyshev'), places=(1, 1))
    births = BirthDeathInteraction('births', target='parents', birth=1.0, offspring=12, placement='neighbor')
    trajectory = System(elements=[parents], interactions=[births]).prepare().simulate(1)
    locations = {tuple(cell) for cell in np.asarray(trajectory.get_agents('parents', 1).locations).tolist()}
    assert len(locations) == 9


def test_live_inherits_and_mutates_the_states_of_offspring():
    """Test that offspring copy the inherited states of their parent, changed by the mutation."""

    def mutate(values, system):
        """Raise the inherited trait by one."""
        return {**values, 'trait': values['trait'] + 1.0}

    parents = _members('parents', 1, trait=2.0, reserve=5.0)
    births = BirthDeathInteraction('births', target='parents', birth=1.0, inherit=('trait',), mutate=mutate)
    trajectory = System(elements=[parents], interactions=[births]).prepare().simulate(1)
    values = trajectory.get_agents('parents', 1).values
    assert values['trait'].tolist() == [2.0, 3.0]
    assert values['reserve'].tolist() == [5.0, 0.0]


def _cycle(life: BirthDeathInteraction, *groups: Element, until: int = 1) -> Trajectory:
    """Return the trajectory of groups of reserve members on a lattice under a life cycle."""
    return System(elements=list(groups), interactions=[life]).prepare().simulate(until, random_state=0)


def test_live_creates_offspring_in_another_group_with_given_values():
    """Test that births into another group create members of it with the values of the parent's function."""
    parents = _members('parents', 2, space=LATTICE, trait=[1.0, 2.0])
    young = _members('young', 1, space=LATTICE, places=(0, 0))
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=1.0,
        group='young',
        values=lambda parent, system: {'trait': parent.trait * 10},
    )
    trajectory = _cycle(life, parents, young)
    assert sorted(trajectory.get_agents('young', 1).values['trait'].tolist()) == [0.0, 10.0, 20.0]
    assert len(trajectory.get_agents('parents', 1).values['trait']) == 2
    assert trajectory.get_values('life.births')[-1] == 2


def test_live_replaces_parents_by_their_offspring():
    """Test that with replacement a parent is removed after its births, also when it has no offspring."""
    parents = _members('parents', 4, space=LATTICE)
    life = BirthDeathInteraction(
        'life',
        target='parents',
        birth=1.0,
        offspring=lambda parent, system: 2 if parent.id % 2 else 0,
        values=lambda parent, system: {'trait': parent.trait + 1},
        replace=True,
    )
    values = _cycle(life, parents).get_agents('parents', 1).values
    assert values['trait'].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_live_keeps_a_share_of_the_split_states():
    """Test that a parent keeps the share `keep` of a split state and its offspring take the rest."""
    parents = _members('parents', 1, space=LATTICE, reserve=8.0)
    life = BirthDeathInteraction('life', target='parents', birth=1.0, offspring=2, split=('reserve',), keep=0.5)
    assert _reserves_of(_cycle(life, parents)) == [2.0, 2.0, 4.0]


def test_live_resets_a_parent_after_its_birth():
    """Test that a parent takes the reset values after it gives birth, and a parent that does not keeps its own."""
    parents = _members('parents', 2, space=LATTICE, reserve=[3.0, 1.0])
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
    parents = _members('parents', 2, space=LATTICE, reserve=[-1.0, 2.0], trait=[7.0, 8.0])
    life = BirthDeathInteraction(
        'life',
        target='parents',
        death=lambda agent, system: agent.reserve < 0,
        renew=True,
        values=lambda agent, system: {'reserve': 1.0},
    )
    trajectory = _cycle(life, parents)
    values = trajectory.get_agents('parents', 1).values
    assert values['reserve'].tolist() == [1.0, 2.0]
    assert values['trait'].tolist() == [7.0, 8.0]
    assert trajectory.get_values('life.deaths')[-1] == 1


def test_live_culls_kin_of_another_group():
    """Test that an agent removes the first member of the other group that is its kin."""
    owners = _members('owners', 2, space=LATTICE, trait=[0.0, 1.0])
    owned = _members('owned', 3, space=LATTICE, trait=[1.0, 0.0, 1.0])
    life = BirthDeathInteraction(
        'life',
        target='owners',
        group='owned',
        cull=lambda owner, system: owner.trait == 1.0,
        kin=lambda owner, member: member.trait == owner.trait,
    )
    assert _cycle(life, owners, owned).get_agents('owned', 1).values['trait'].tolist() == [0.0, 1.0]


def test_live_brings_arrivals_at_placed_locations():
    """Test that a Poisson number of arrivals join the group at the placement's locations with drawn values."""
    members = _members('members', 1, space=LATTICE, places=(2, 1))
    life = BirthDeathInteraction(
        'life',
        target='members',
        arrival=lambda system: 50.0,
        placement=lambda parent, system: (2, 1),
        values=lambda parent, system: {'trait': float(system.rng.integers(5))},
    )
    trajectory = _cycle(life, members, until=4)
    snapshot = trajectory.get_agents('members', 4)
    assert (len(snapshot.values['trait']) - 1) / 4 == pytest.approx(50.0, rel=0.2)
    assert {tuple(cell) for cell in np.asarray(snapshot.locations).tolist()} == {(2, 1)}
    assert set(snapshot.values['trait'].tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}


def _reserves_of(trajectory: Trajectory) -> list[float]:
    """Return the sorted reserves of the parents at the first frame."""
    return sorted(trajectory.get_agents('parents', 1).values['reserve'].tolist())


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
    parents = _members('parents', 4, space=LATTICE, trait=1.0)
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
    system = System(
        elements=[parents],
        interactions=[life],
        solver=MesaSolver(phases=[PhaseSpec('default', activation='visits', visits=0.5)]),
    )
    trajectory = system.prepare().simulate(1, random_state=0)
    assert len(trajectory.get_agents('parents', 1).ids) == 4


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
