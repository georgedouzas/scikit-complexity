"""Test the `_life_cycle` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import LifeCycleInteraction, TransferInteraction
from skcomplex.core.laws import Sink, Source
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


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
        amount=lambda cell, forager, model: min(cell.resource, uptake),
        legs=((Source(), 'reserve', lambda cell, forager, model: gain * min(cell.resource, uptake)),),
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


def _life(threshold: float = 0.0, probability: float = 0.0) -> LifeCycleInteraction:
    """Return the death of foragers below zero reserve and their split above a threshold with a probability."""
    return LifeCycleInteraction(
        'life',
        target='foragers',
        death=lambda forager, model: forager.reserve < 0,
        birth=lambda forager, model: probability if forager.reserve >= threshold else 0.0,
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
    system = System(elements=[cells, foragers], interactions=interactions, space=LatticeSpace((1, 1)))
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


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
    system = System(
        elements=[cells, foragers],
        interactions=[_feeding(uptake=10.0, gain=1.0)],
        space=NetworkSpace(nx.path_graph(2)),
    )
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert _reserves(trajectory, 1) == [4.0]
    assert trajectory.agents('cells', 1).values['resource'].tolist() == [1.0, 0.0]


def test_live_rate_draws_a_poisson_number_of_births():
    """Test that births at a rate follow a Poisson number of that mean, each with the offspring count."""
    parents = GroupElement('parents', element=ReserveElement, size=400, variable=True, capacity=4000)
    births = LifeCycleInteraction('births', target='parents', rate=1.5, offspring=2)
    trajectory = Simulation(System(elements=[parents], interactions=[births])).check().prepare().run(until=1)
    assert len(trajectory.trajectory_.agents('parents', 1).values['reserve']) / 400 - 1 == pytest.approx(3.0, rel=0.1)


def test_live_places_offspring_in_free_neighboring_cells():
    """Test that offspring placed in neighbouring cells take free cells only, and fewer are born when none is free."""
    parents = GroupElement('parents', element=ReserveElement, size=1, location=[(1, 1)], variable=True)
    births = LifeCycleInteraction('births', target='parents', birth=1.0, offspring=12, placement='neighbor')
    space = LatticeSpace((3, 3), neighborhood='moore')
    trajectory = (
        Simulation(System(elements=[parents], interactions=[births], space=space)).check().prepare().run(until=1)
    )
    locations = {tuple(cell) for cell in np.asarray(trajectory.trajectory_.agents('parents', 1).locations).tolist()}
    assert len(locations) == 9


def test_live_inherits_and_mutates_the_states_of_offspring():
    """Test that offspring copy the inherited states of their parent, changed by the mutation."""

    def mutate(values, model):
        """Raise the inherited trait by one."""
        return {**values, 'trait': values['trait'] + 1.0}

    parents = GroupElement(
        'parents',
        element=ReserveElement,
        size=1,
        values={'trait': 2.0, 'reserve': 5.0},
        variable=True,
    )
    births = LifeCycleInteraction('births', target='parents', birth=1.0, inherit=('trait',), mutate=mutate)
    trajectory = Simulation(System(elements=[parents], interactions=[births])).check().prepare().run(until=1)
    values = trajectory.trajectory_.agents('parents', 1).values
    assert values['trait'].tolist() == [2.0, 3.0]
    assert values['reserve'].tolist() == [5.0, 0.0]


@pytest.mark.parametrize(
    'arguments',
    [{'placement': 'far'}, {'birth': 0.5, 'rate': 1.0}, {'death': 'old'}, {'mutate': 3}],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown placement, two kinds of birth, and values that are not numbers or functions are rejected."""
    with pytest.raises(ValidationError):
        LifeCycleInteraction('life', target='foragers', **arguments).check()
