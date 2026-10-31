"""Test the `_movement` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import MovementInteraction, TransferInteraction
from skcomplex.core.laws import Sink
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class BudgetElement(Element):
    """Walker with a level that its steps cost."""

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


def _lattice_walk(until: int, walk: MovementInteraction, size: int = 4) -> Trajectory:
    """Return the trajectory of walkers that start at the middle of a 9 by 9 lattice."""
    walkers = GroupElement('walkers', element=BudgetElement, size=size, location=[(4, 4)] * size)
    space = LatticeSpace((9, 9), neighborhood='von_neumann')
    return _run(System(elements=[walkers], interactions=[walk], space=space), until)


def _cost(amount: float) -> TransferInteraction:
    """Return the transfer of an amount of the level of a walker to the environment."""
    return TransferInteraction(
        'cost',
        source='walkers',
        target='walkers',
        state='level',
        into=Sink(),
        amount=amount,
        form='agent',
    )


def test_move_gaussian_steps_spread_as_diffusion():
    """Test that 10,000 Gaussian walkers of variance `2 D` per step have the mean squared displacement `4 D t`."""
    D, until, size = 0.5, 20, 1000.0
    middle = (size / 2, size / 2)
    walkers = GroupElement('walkers', element=Element, size=10_000, location=[middle] * 10_000)
    walk = MovementInteraction('walk', target='walkers', step='gaussian', scale=np.sqrt(2 * D))
    space = ContinuousSpace(extent=(size, size), boundary='periodic')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), until)
    displacements = np.asarray(trajectory.agents('walkers', until).locations, dtype=float) - middle
    assert (displacements**2).sum(axis=1).mean() == pytest.approx(4 * D * until, rel=0.05)


def test_move_levy_steps_are_at_least_the_scale():
    """Test that every Lévy step moves a walker by at least the scale."""
    middle = (50.0, 50.0)
    walkers = GroupElement('walkers', element=Element, size=200, location=[middle] * 200)
    walk = MovementInteraction('walk', target='walkers', step='levy', scale=0.5, exponent=1.5)
    space = ContinuousSpace(extent=(100.0, 100.0), boundary='periodic')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 1)
    lengths = np.linalg.norm(np.asarray(trajectory.agents('walkers', 1).locations, dtype=float) - middle, axis=1)
    assert lengths.min() >= 0.5


def test_move_neighbor_steps_move_to_adjacent_cells():
    """Test that neighbour steps on a lattice move every walker to one of the four adjacent cells."""
    trajectory = _lattice_walk(5, MovementInteraction('walk', target='walkers'))
    locations = np.asarray([trajectory.agents('walkers', frame).locations for frame in range(6)], dtype=int)
    steps = np.abs(np.diff(locations, axis=0)).sum(axis=-1)
    assert np.all(steps == 1)


def test_move_neighbor_steps_follow_the_edges_of_a_network():
    """Test that neighbour steps on a network move every walker along an edge of the graph."""
    graph = nx.path_graph(6)
    walkers = GroupElement('walkers', element=Element, size=3, location=[2, 3, 4])
    walk = MovementInteraction('walk', target='walkers')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=NetworkSpace(graph)), 10)
    nodes = np.asarray([trajectory.agents('walkers', frame).locations for frame in range(11)], dtype=int)
    assert np.all(np.abs(np.diff(nodes, axis=0)) == 1)


def test_move_charges_a_transfer_for_every_step():
    """Test that every step pays the amount of a transfer from the level of the walker."""
    trajectory = _lattice_walk(3, MovementInteraction('walk', target='walkers', cost=_cost(1.5)))
    assert trajectory.agents('walkers', 3).values['level'].tolist() == [5.5] * 4


def test_move_charges_a_function_for_every_step():
    """Test that every step runs a cost function of the walker and the model view."""

    def charge(walker, model):
        """Halve the level of a walker."""
        walker.level = walker.level / 2

    trajectory = _lattice_walk(2, MovementInteraction('walk', target='walkers', cost=charge))
    assert trajectory.agents('walkers', 2).values['level'].tolist() == [2.5] * 4


def test_move_without_probability_keeps_walkers_in_place():
    """Test that walkers whose step probability is zero never move or pay."""
    trajectory = _lattice_walk(5, MovementInteraction('walk', target='walkers', probability=0.0, cost=_cost(1.0)))
    assert np.all(np.asarray(trajectory.agents('walkers', 5).locations) == 4)
    assert trajectory.agents('walkers', 5).values['level'].tolist() == [10.0] * 4


def test_move_rate_draws_a_poisson_number_of_steps():
    """Test that walkers with a rate take a Poisson number of steps of that mean, as their spent level counts."""
    walkers = GroupElement('walkers', element=BudgetElement, size=500, values={'level': 0.0}, location=[(4, 4)] * 500)
    walk = MovementInteraction('walk', target='walkers', rate=2.0, cost=_cost(1.0))
    space = LatticeSpace((9, 9), neighborhood='von_neumann')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 10)
    steps = -trajectory.agents('walkers', 10).values['level']
    assert steps.mean() == pytest.approx(20.0, rel=0.05)
    assert steps.var() == pytest.approx(20.0, rel=0.2)


def test_move_condition_moves_only_the_agents_where_it_holds():
    """Test that only the walkers whose condition holds move."""
    walkers = GroupElement(
        'walkers',
        element=BudgetElement,
        size=2,
        values={'level': [1.0, 0.0]},
        location=[(4, 4)] * 2,
    )
    walk = MovementInteraction('walk', target='walkers', condition=lambda walker, model: walker.level > 0)
    space = LatticeSpace((9, 9), neighborhood='von_neumann')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 1)
    locations = np.asarray(trajectory.agents('walkers', 1).locations)
    assert np.abs(locations - 4).sum(axis=1).tolist() == [1, 0]


def test_move_accept_rejects_every_step_of_zero_probability():
    """Test that an acceptance of zero keeps every walker in place."""
    walk = MovementInteraction('walk', target='walkers', accept=lambda walker, cell, model: 0.0)
    trajectory = _lattice_walk(3, walk)
    assert np.all(np.asarray(trajectory.agents('walkers', 3).locations) == 4)


def test_move_best_steps_climb_a_score():
    """Test that best steps move walkers up a score to its maximum and keep them there."""
    walk = MovementInteraction('walk', target='walkers', step='best', score=lambda walker, cell, model: cell[0])
    trajectory = _lattice_walk(6, walk, size=1)
    assert np.asarray(trajectory.agents('walkers', 4).locations).tolist() == [[8, 4]]


def test_move_exclusive_steps_never_share_a_cell():
    """Test that exclusive walkers on a crowded lattice never stand in one cell."""
    cells = [(row, column) for row in range(3) for column in range(3)][:6]
    walkers = GroupElement('walkers', element=Element, size=6, location=cells)
    walk = MovementInteraction('walk', target='walkers', exclusive=True)
    space = LatticeSpace((3, 3), neighborhood='moore')
    trajectory = _run(System(elements=[walkers], interactions=[walk], space=space), 10)
    for frame in range(11):
        locations = {tuple(cell) for cell in np.asarray(trajectory.agents('walkers', frame).locations).tolist()}
        assert len(locations) == 6


def test_move_function_steps_go_to_the_returned_location():
    """Test that a step function moves every walker to the location it returns."""
    walk = MovementInteraction('walk', target='walkers', step=lambda walker, model: (0, 0))
    trajectory = _lattice_walk(1, walk)
    assert np.asarray(trajectory.agents('walkers', 1).locations).tolist() == [[0, 0]] * 4


@pytest.mark.parametrize(
    'arguments',
    [
        {'step': 'cauchy'},
        {'rate': 0.0},
        {'rate': True},
        {'step': 'best'},
        {'condition': 3},
        {'cost': 'level'},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown step, a bad rate, a best step without score, and bad functions are rejected."""
    with pytest.raises(ValidationError):
        MovementInteraction('walk', target='walkers', **arguments).check()
