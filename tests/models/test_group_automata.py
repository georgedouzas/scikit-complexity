"""Test the automata on groups: the named groups, their balls and quotients, and the group automata models."""

import numpy as np
import pytest

from skcomplex.core.analysis import Subshift
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.space import GroupSpace, Memory, automaton_group, parse_word
from skcomplex.core.viz import CayleyVisual
from skcomplex.models import catalog, create
from skcomplex.models.computer_science.automata import (
    baumslag_solitar_automaton,
    free_group_automaton,
    grigorchuk_group,
    heisenberg,
    heisenberg_automaton,
    lamplighter_automaton,
    linear_ring_automaton,
    subshifts,
)

GRIGORCHUK_BALLS = [1, 5, 11, 23, 40, 68, 108, 176, 271]


def _run(system, until):
    """Return a simulation of a system."""
    return Simulation(system, dt=1, random_state=0).run(until=until)


def _majority_holds(space, memory, before, after):
    """Check that every interior cell takes the majority value of its memory set."""
    table = space.index_table(memory)
    interior = (table < space.size).all(axis=1)
    expected = before[table[interior]].sum(axis=1) > len(memory) / 2
    return bool(np.array_equal(after[interior], expected))


def _site_of(space, word):
    """Return the site reached from the identity along the letters of a word."""
    site = 0
    for letter in parse_word(word, space.group.generators):
        site = int(space.generator_table[site, 2 * (abs(letter) - 1) + (letter < 0)])
    return site


def test_grigorchuk_group_is_an_automaton_group():
    """Test that the Grigorchuk group of the library is an automaton group of the core with generators a to d."""
    group = grigorchuk_group()
    recursion = {
        'a': ((None, None), (1, 0)),
        'b': (('a', 'c'), None),
        'c': (('a', 'd'), None),
        'd': ((None, 'b'), None),
    }
    assert group == automaton_group(recursion)
    assert group.generators == ('a', 'b', 'c', 'd')


def test_grigorchuk_growth():
    """Test that the balls of radii 0 to 5 hold 1 to 68 sites, all distinct elements as their level-12 actions show."""
    ball = GroupSpace(grigorchuk_group(), ball=5)
    assert ball.ball_sizes() == GRIGORCHUK_BALLS[:6]
    action = grigorchuk_group().level_quotient(12).word_problem()
    keys = set()
    for word in ball.words:
        element = action.identity()
        for letter in parse_word(word, ball.group.generators):
            element = action.multiply(element, letter)
        keys.add(action.key(element))
    assert len(keys) == ball.size


def test_grigorchuk_level_quotients_run_periodic_rules():
    """Test that the quotients of levels 1 to 4 hold 2, 8, 128, and 4096 sites and every cell reads inside them."""
    sizes = []
    for level in (1, 2, 3, 4):
        space = GroupSpace(grigorchuk_group(), quotient=level)
        sizes.append(space.size)
        assert (space.index_table(Memory(('1', 'a', 'b', 'c', 'd'))) < space.size).all()
    assert sizes == [2, 8, 128, 4096]


def test_grigorchuk_automaton_runs_the_majority_rule():
    """Test that the model runs the majority of {1, a, b, c, d} on a ball of radius 8 with the known ball sizes."""
    simulation = _run(create('grigorchuk_automaton'), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_['cells.alive']
    assert space.size == 271
    assert space.ball_sizes() == GRIGORCHUK_BALLS
    assert _majority_holds(space, Memory(('1', 'a', 'b', 'c', 'd')), alive[0], alive[1])
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

    visual = CayleyVisual(low=0, high=1)
    trajectory = simulation.trajectory_
    for index in (0, len(trajectory.times) - 1):
        frame = Frame(index, trajectory.times[index], trajectory.states[index], simulation.parameters_, problem)
        visual.log(Recording(), problem.elements[0], frame)
    nodes = [archetype for static, archetype in logged if not static]
    positions = np.asarray(nodes[0].positions.as_arrow_array().to_pylist())
    assert positions.shape == (271, 2)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1) / 40.0, space.lengths)
    first, last = (np.asarray(node.colors.as_arrow_array().to_pylist()) for node in nodes)
    assert not np.array_equal(first, last)
    assert sum(static for static, _ in logged) == 1


def test_free_group_automaton_runs_the_majority_rule():
    """Test that every interior cell of the free group ball takes the majority of itself and its four neighbours."""
    simulation = _run(free_group_automaton(radius=6), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_['cells.alive']
    assert space.size == 1457
    assert _majority_holds(space, Memory(('1', 'a', 'a^-1', 'b', 'b^-1')), alive[0], alive[1])


@pytest.mark.parametrize(('factory', 'size'), [(heisenberg_automaton, 17), (baumslag_solitar_automaton, 17)])
def test_majority_automata_on_balls(factory, size):
    """Test that the majority automata on the Heisenberg and Baumslag-Solitar balls of radius 2 run the rule."""
    simulation = _run(factory(radius=2), 1)
    space = simulation.system.elements[0].space
    alive = simulation.trajectory_['cells.alive']
    assert space.size == size
    assert _majority_holds(space, Memory.generators(space.group), alive[0], alive[1])


def test_heisenberg_commutator_is_central():
    """Test that the commutator of the Heisenberg generators commutes with both generators."""
    space = GroupSpace(heisenberg(), ball=6)
    commutator = 'a*b*a^-1*b^-1'
    for generator in ('a', 'b'):
        assert _site_of(space, f'{commutator}*{generator}') == _site_of(space, f'{generator}*{commutator}')


def test_lamplighter_automaton_is_linear():
    """Test that the parity automaton on the lamplighter ball maps a sum of configurations to the sum of images."""
    system = lamplighter_automaton(radius=5)
    space = system.elements[0].space
    table = space.index_table(Memory(('1', 'a', 'b')))
    interior = (table < space.size).all(axis=1)
    rng = np.random.default_rng(0)
    first, second = rng.integers(2, size=(2, space.size))
    images = [np.append(cells, 0)[table].sum(axis=1) % 2 for cells in (first, second, (first + second) % 2)]
    np.testing.assert_array_equal(((images[0] + images[1]) % 2)[interior], images[2][interior])
    alive = _run(system, 1).trajectory_['cells.alive']
    np.testing.assert_array_equal(alive[1][interior], (np.append(alive[0], 0)[table].sum(axis=1) % 2)[interior])


def test_linear_ring_automaton_spreads_binomials():
    """Test that the Laplacian over GF(3) from one cell gives the coefficients of (x + 1/x - 2)^t modulo 3."""
    simulation = _run(linear_ring_automaton(11, modulus=3), 2)
    space = simulation.system.elements[0].space
    values = simulation.trajectory_['cells.value'][-1]
    expected = {'1': 6 % 3, 'a': -4 % 3, 'a^-1': -4 % 3, 'a*a': 1, 'a^-1*a^-1': 1}
    for word, value in expected.items():
        assert values[space.words.index(word)] == value


def test_subshifts_stay_admissible():
    """Test that the Markov chain on the golden mean shift never puts two ones side by side and fills about a fifth."""
    simulation = _run(subshifts(100), 30)
    space = simulation.system.elements[0].space
    golden = Subshift(Memory(('1', 'a')), [(1, 1)])
    alive = simulation.trajectory_['cells.alive']
    assert all(golden.is_admissible(row.astype(int), space) for row in alive)
    assert 0.1 < alive[-10:].mean() < 0.4
