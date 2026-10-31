"""Test the `_random_walk` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RandomWalkInteraction
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class BudgetElement(Element):
    """Walker with a budget that its steps cost."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level


def _run(system: System, until: int) -> Trajectory:
    """Return the trajectory of a system run to a time."""
    return Simulation(system, random_state=0).check().prepare().run(until=until).trajectory_


def _lattice_walk(until: int, walk: RandomWalkInteraction) -> Trajectory:
    """Return the trajectory of four walkers that start at the middle of a 9 by 9 lattice."""
    walkers = GroupElement('walkers', element=BudgetElement, size=4, location=[(4, 4)] * 4)
    space = LatticeSpace((9, 9), neighborhood='von_neumann')
    return _run(System(elements=[walkers], interactions=[walk], space=space), until)


def test_walk_gaussian_steps_spread_as_diffusion():
    """Test that 10,000 Gaussian walkers of variance `2 D` per step have the mean squared displacement `4 D t`."""
    D, until, size = 0.5, 20, 1000.0
    middle = (size / 2, size / 2)
    walkers = GroupElement('walkers', element=Element, size=10_000, location=[middle] * 10_000)
    walk = RandomWalkInteraction('walk', walkers='walkers', step='gaussian', scale=np.sqrt(2 * D))
    space = ContinuousSpace(extent=(size, size), boundary='periodic')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), until)
    displacements = np.asarray(trajectory.agents('walkers', until).locations, dtype=float) - middle
    assert (displacements**2).sum(axis=1).mean() == pytest.approx(4 * D * until, rel=0.05)


def test_walk_levy_steps_are_at_least_the_scale():
    """Test that every Lévy step moves a walker by at least the scale."""
    middle = (50.0, 50.0)
    walkers = GroupElement('walkers', element=Element, size=200, location=[middle] * 200)
    walk = RandomWalkInteraction('walk', walkers='walkers', step='levy', scale=0.5, exponent=1.5)
    space = ContinuousSpace(extent=(100.0, 100.0), boundary='periodic')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 1)
    lengths = np.linalg.norm(np.asarray(trajectory.agents('walkers', 1).locations, dtype=float) - middle, axis=1)
    assert lengths.min() >= 0.5


def test_walk_neighbor_steps_move_to_adjacent_cells():
    """Test that neighbour steps on a lattice move every walker to one of the four adjacent cells."""
    trajectory = _lattice_walk(5, RandomWalkInteraction('walk', walkers='walkers'))
    locations = np.asarray([trajectory.agents('walkers', frame).locations for frame in range(6)], dtype=int)
    steps = np.abs(np.diff(locations, axis=0)).sum(axis=-1)
    assert np.all(steps == 1)


def test_walk_neighbor_steps_follow_the_edges_of_a_network():
    """Test that neighbour steps on a network move every walker along an edge of the graph."""
    graph = nx.path_graph(6)
    walkers = GroupElement('walkers', element=Element, size=3, location=[2, 3, 4])
    walk = RandomWalkInteraction('walk', walkers='walkers')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=NetworkSpace(graph)), 10)
    nodes = np.asarray([trajectory.agents('walkers', frame).locations for frame in range(11)], dtype=int)
    assert np.all(np.abs(np.diff(nodes, axis=0)) == 1)


def test_walk_charges_the_cost_of_every_step_to_the_budget():
    """Test that every step subtracts the cost from the named budget state."""
    trajectory = _lattice_walk(3, RandomWalkInteraction('walk', walkers='walkers', cost=1.5, budget='level'))
    assert trajectory.agents('walkers', 3).values['level'].tolist() == [5.5] * 4


def test_walk_without_probability_keeps_walkers_in_place():
    """Test that walkers whose step probability is zero never move."""
    trajectory = _lattice_walk(
        5,
        RandomWalkInteraction('walk', walkers='walkers', probability=0.0, cost=1.0, budget='level'),
    )
    assert np.all(np.asarray(trajectory.agents('walkers', 5).locations) == 4)
    assert trajectory.agents('walkers', 5).values['level'].tolist() == [10.0] * 4


def test_walk_rate_draws_a_poisson_number_of_steps():
    """Test that walkers with a rate take a Poisson number of steps of that mean, as their spent budget counts."""
    walkers = GroupElement('walkers', element=BudgetElement, size=500, values={'level': 0.0}, location=[(4, 4)] * 500)
    walk = RandomWalkInteraction('walk', walkers='walkers', rate=2.0, cost=1.0, budget='level')
    space = LatticeSpace((9, 9), neighborhood='von_neumann')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 10)
    steps = -trajectory.agents('walkers', 10).values['level']
    assert steps.mean() == pytest.approx(20.0, rel=0.05)
    assert steps.var() == pytest.approx(20.0, rel=0.2)


@pytest.mark.parametrize(
    'arguments',
    [{'step': 'cauchy'}, {'rate': 0.0}, {'rate': True}, {'budget': 3}],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown step, a rate that is not positive, and a budget that is not a name are rejected."""
    with pytest.raises(ValidationError):
        RandomWalkInteraction('walk', walkers='walkers', **arguments).check()
