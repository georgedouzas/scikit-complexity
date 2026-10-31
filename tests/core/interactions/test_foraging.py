"""Test the `_foraging` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import ForagingInteraction
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
    """Member that holds a reserve."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        reserve: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.reserve = reserve


def _foraging(
    uptake: float = 1.0,
    gain: float = 1.0,
    metabolism: float = 0.0,
    threshold: float = 0.0,
    probability: float = 0.0,
) -> ForagingInteraction:
    """Return the foraging of the group `foragers` from the group `cells`."""
    return ForagingInteraction(
        'foraging',
        foragers='foragers',
        resources='cells',
        uptake=uptake,
        gain=gain,
        metabolism=metabolism,
        threshold=threshold,
        probability=probability,
    )


def _forage(until: int, foraging: ForagingInteraction, resource: float, reserve: float, size: int = 1) -> Trajectory:
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
    system = System(elements=[cells, foragers], interactions=[foraging], space=LatticeSpace((1, 1)))
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _reserves(trajectory: Trajectory, frame: int) -> list[float]:
    """Return the reserves of the foragers alive at a frame."""
    return trajectory.agents('foragers', frame).values['reserve'].tolist()


def test_forage_gains_the_resource_it_takes():
    """Test that a forager takes at most its uptake from its cell and gains it times the gain."""
    trajectory = _forage(2, _foraging(uptake=2.0, gain=0.5), resource=3.0, reserve=0.0)
    assert trajectory.agents('cells', 2).values['resource'].tolist() == [0.0]
    assert _reserves(trajectory, 1) == [1.0]
    assert _reserves(trajectory, 2) == [1.5]


def test_forage_shares_a_scarce_resource_between_foragers():
    """Test that two foragers of one cell take no more than the cell holds."""
    trajectory = _forage(1, _foraging(uptake=1.0), resource=1.5, reserve=0.0, size=2)
    assert sorted(_reserves(trajectory, 1)) == [0.5, 1.0]


def test_forage_pays_the_metabolism():
    """Test that a forager without food loses its metabolism every step."""
    trajectory = _forage(3, _foraging(metabolism=1.0), resource=0.0, reserve=5.0)
    assert _reserves(trajectory, 3) == [2.0]


def test_forage_removes_foragers_below_zero_reserve():
    """Test that a forager survives at zero reserve and dies when its reserve falls below zero."""
    trajectory = _forage(2, _foraging(metabolism=1.0), resource=0.0, reserve=1.0)
    assert _reserves(trajectory, 1) == [0.0]
    assert _reserves(trajectory, 2) == []


def test_forage_splits_the_reserve_above_the_threshold():
    """Test that a forager at its threshold splits its reserve with an offspring, and one below it does not."""
    split = _forage(1, _foraging(metabolism=1.0, threshold=6.0, probability=1.0), resource=0.0, reserve=8.0)
    kept = _forage(1, _foraging(metabolism=1.0, threshold=6.0, probability=1.0), resource=0.0, reserve=6.0)
    assert _reserves(split, 1) == [3.5, 3.5]
    assert _reserves(kept, 1) == [5.0]


def test_forage_feeds_foragers_on_the_nodes_of_a_network():
    """Test that foragers on a network take the resource of their own node."""
    cells = GroupElement('cells', element=ResourceElement, size=2, values={'resource': [1.0, 4.0]}, location=[0, 1])
    foragers = GroupElement('foragers', element=ReserveElement, size=1, location=[1])
    foraging = ForagingInteraction('foraging', foragers='foragers', resources='cells', uptake=10.0)
    system = System(elements=[cells, foragers], interactions=[foraging], space=NetworkSpace(nx.path_graph(2)))
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert _reserves(trajectory, 1) == [4.0]
    assert trajectory.agents('cells', 1).values['resource'].tolist() == [1.0, 0.0]


@pytest.mark.parametrize('name', ['resource', 'reserve'])
def test_check_rejects_a_state_name_that_is_not_a_string(name):
    """Test that a resource or a reserve that is not a string is rejected."""
    with pytest.raises(ValidationError):
        ForagingInteraction('foraging', foragers='foragers', resources='cells', **{name: 1}).check()
