"""Test the `_group_automata` module of `skcomplex.models.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.spaces import GeneratorNeighborhood, LatticeNeighborhood, MemoryNeighborhood
from skcomplex.engine.problems import FrameSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import CayleyVisual
from skcomplex.models import catalog, create
from skcomplex.models.computer_science.automata import (
    Subshift,
    baumslag_solitar_automaton,
    free_group_automaton,
    heisenberg_automaton,
    lamplighter_automaton,
    linear_ring_automaton,
    subshifts,
)

from .conftest import GRIGORCHUK_BALLS


def _run(system, until):
    """Return a simulation of a system."""
    return Simulation(system, dt=1, random_state=0).check().run(until=until)


def _majority_holds(space, memory, before, after):
    """Check that every interior cell takes the majority value of its memory set."""
    table = space._index_table(memory)
    interior = (table < space.size_).all(axis=1)
    expected = before[table[interior]].sum(axis=1) > table.shape[1] / 2
    return bool(np.array_equal(after[interior], expected))


def test_grigorchuk_automaton_runs_the_majority_rule():
    """Test that the model runs the majority of {1, a, b, c, d} on a ball of radius 8 with the known ball sizes."""
    simulation = _run(create('grigorchuk_automaton'), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_.values('cells.alive')
    assert space.size_ == 271
    assert space.ball_sizes_ == GRIGORCHUK_BALLS
    assert _majority_holds(space, MemoryNeighborhood(('1', 'a', 'b', 'c', 'd')), alive[0], alive[1])
    assert (
        next(entry for entry in catalog() if entry.name == 'grigorchuk_automaton').field == 'computer_science/automata'
    )


def test_grigorchuk_automaton_in_the_viewer():
    """Test that the Cayley visual logs 271 nodes on rings by word length, colored by the states of every frame."""
    simulation = _run(create('grigorchuk_automaton'), 3)
    problem = simulation.problem_
    space = simulation.system.elements[0].space
    logged = []

    class Recording:
        """Recording stand-in that keeps what it receives."""

        def log(self, path, archetype, static=False):
            """Keep a logged entity."""
            logged.append((static, archetype))

    visual = CayleyVisual(low=0, high=1, path=problem.elements[0].path, recording=Recording()).check().prepare(problem)
    trajectory = simulation.trajectory_
    for index in (0, len(trajectory.times) - 1):
        frame = FrameSpec(index, trajectory.times[index], trajectory.states[index], simulation.parameters_, problem)
        visual.run(frame)
    nodes = [archetype for static, archetype in logged if not static]
    positions = np.asarray(nodes[0].positions.as_arrow_array().to_pylist())
    assert positions.shape == (271, 2)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1) / 40.0, space.lengths_)
    first, last = (np.asarray(node.colors.as_arrow_array().to_pylist()) for node in nodes)
    assert not np.array_equal(first, last)
    assert sum(static for static, _ in logged) == 1


def test_free_group_automaton_runs_the_majority_rule():
    """Test that every interior cell of the free group ball takes the majority of itself and its four neighbours."""
    simulation = _run(free_group_automaton(radius=6), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_.values('cells.alive')
    assert space.size_ == 1457
    assert _majority_holds(space, MemoryNeighborhood(('1', 'a', 'a^-1', 'b', 'b^-1')), alive[0], alive[1])


@pytest.mark.parametrize(('factory', 'size'), [(heisenberg_automaton, 17), (baumslag_solitar_automaton, 17)])
def test_majority_automata_on_balls(factory, size):
    """Test that the majority automata on the Heisenberg and Baumslag-Solitar balls of radius 2 run the rule."""
    simulation = _run(factory(radius=2), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_.values('cells.alive')
    assert space.size_ == size
    assert _majority_holds(space, GeneratorNeighborhood(space.generators_), alive[0], alive[1])


def test_lamplighter_automaton_is_linear():
    """Test that the parity automaton on the lamplighter ball maps a sum of configurations to the sum of images."""
    system = lamplighter_automaton(radius=5)
    space = system.elements[0].space
    table = space._index_table(MemoryNeighborhood(('1', 'a', 'b')))
    interior = (table < space.size_).all(axis=1)
    rng = np.random.default_rng(0)
    first, second = rng.integers(2, size=(2, space.size_))
    images = [np.append(cells, 0)[table].sum(axis=1) % 2 for cells in (first, second, (first + second) % 2)]
    np.testing.assert_array_equal(((images[0] + images[1]) % 2)[interior], images[2][interior])
    alive = _run(system, 1).trajectory_.values('cells.alive')
    np.testing.assert_array_equal(alive[1][interior], (np.append(alive[0], 0)[table].sum(axis=1) % 2)[interior])


def test_linear_ring_automaton_spreads_binomials():
    """Test that the Laplacian over GF(3) from one cell gives the coefficients of (x + 1/x - 2)^t modulo 3."""
    values = _run(linear_ring_automaton(11, modulus=3), 2).trajectory_.values('cells.value')[-1]
    np.testing.assert_array_equal(values, [0, 0, 0, 1, 2, 0, 2, 1, 0, 0, 0])


def test_subshifts_stay_admissible():
    """Test that the Markov chain on the golden mean shift never puts two ones side by side and fills about a fifth."""
    simulation = _run(subshifts(100), 30)
    space = simulation.system.elements[0].space
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    alive = simulation.trajectory_.values('cells.alive')
    assert all(golden.is_admissible(row.astype(int), space) for row in alive)
    assert 0.1 < alive[-10:].mean() < 0.4
