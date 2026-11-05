"""Test the `_automaton` module of `skcomplex.models.computer_science.automata`."""

import hashlib

import numpy as np
import pytest

from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace, MooreNeighborhood
from skcomplex.errors import ValidationError
from skcomplex.models.computer_science.automata import Automaton, Wolfram
from skcomplex.specs import OuterTotalisticSpec, TableSpec, TotalisticSpec


def test_shift_moves_every_cell_one_site():
    """Test that the shift reads the right neighbour, so every step rolls the ring to the left."""
    shift = Automaton(LatticeSpace(6), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    configuration = np.array([1, 0, 0, 1, 1, 0])
    runs = shift.run(configuration, 3)
    assert runs.shape == (4, 6)
    for step, row in enumerate(runs):
        np.testing.assert_array_equal(row, np.roll(configuration, -step))


def test_rule_code_is_read_through_its_table():
    """Test that a Wolfram code gives rule 90, the sum of the two neighbours modulo 2."""
    rule90 = Automaton(LatticeSpace(7), BoxNeighborhood(), Wolfram(90))
    configuration = np.array([0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(rule90.step(configuration), [0, 0, 1, 0, 1, 0, 0])
    assert rule90.local_map().size == 8


def test_outer_totalistic_rule_reads_the_cell_first():
    """Test that Life reads the cell and its eight neighbours, and that a blinker turns."""
    life = Automaton(LatticeSpace((5, 5)), MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))
    assert life.table().shape == (25, 9)
    blinker = np.zeros((5, 5), dtype=int)
    blinker[2, 1:4] = 1
    turned = life.step(blinker.ravel()).reshape(5, 5)
    np.testing.assert_array_equal(turned, blinker.T)


def test_rule_values_outside_the_states_raise():
    """Test that a rule returning a value outside 0 to states - 1 is rejected."""
    wrong = Automaton(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total))
    with pytest.raises(ValidationError, match='outside the integers 0 to 1'):
        wrong.local_map()


def test_reference_values_of_life():
    """Test that Life keeps its recorded step, run, and local map."""
    life = Automaton(LatticeSpace((5, 5)), MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))
    glider = np.zeros((5, 5), dtype=int)
    glider[[0, 1, 2, 2, 2], [1, 2, 0, 1, 2]] = 1
    assert np.flatnonzero(life.step(glider.ravel())).tolist() == [5, 7, 11, 12, 16]
    assert np.flatnonzero(life.run(glider.ravel(), 4)[-1]).tolist() == [7, 13, 16, 17, 18]
    local_map = life.local_map()
    assert local_map.size == 512
    assert int(local_map.sum()) == 140
    assert hashlib.sha256(local_map.tobytes()).hexdigest()[:16] == '9c774357e78bfee0'


def test_reference_values_of_elementary_rules():
    """Test that rules 90 and 30 keep their recorded runs and local maps, on one row and on a batch of rows."""
    rule90 = Automaton(LatticeSpace(11), BoxNeighborhood(), Wolfram(90))
    start = np.zeros(11, dtype=int)
    start[5] = 1
    assert [np.flatnonzero(row).tolist() for row in rule90.run(start, 4)] == [[5], [4, 6], [3, 7], [2, 4, 6, 8], [1, 9]]
    assert rule90.local_map().tolist() == [0, 1, 0, 1, 1, 0, 1, 0]
    batch = rule90.step(np.eye(11, dtype=int)[:2])
    assert [np.flatnonzero(row).tolist() for row in batch] == [[1, 10], [0, 2]]
    rule30 = Automaton(LatticeSpace(41), BoxNeighborhood(), Wolfram(30))
    assert rule30.local_map().tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    last = rule30.run(np.eye(41, dtype=int)[20], 10)[-1]
    assert np.flatnonzero(last).tolist() == [10, 11, 14, 19, 21, 22, 23, 24, 26, 27, 30]


def test_reference_values_of_other_rules():
    """Test that fixed edges with a fill, totalistic, function, and three-state rules keep their recorded steps."""
    fixed = Automaton(LatticeSpace(5, boundary='fixed', fill=1), BoxNeighborhood(), Wolfram(90))
    assert fixed.step([0, 0, 1, 0, 0]).tolist() == [1, 1, 0, 1, 1]
    parity = Automaton(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total % 2))
    assert parity.step([0, 0, 1, 0, 0]).tolist() == [0, 1, 1, 1, 0]
    function = Automaton(LatticeSpace(5), BoxNeighborhood(), lambda left, _, right: (left + right) % 2)
    assert function.step([0, 0, 1, 0, 0]).tolist() == [0, 1, 0, 1, 0]
    three = TotalisticSpec(lambda total: total % 3, states=3)
    assert Automaton(LatticeSpace(5), BoxNeighborhood(), three, 3).step([0, 2, 1, 0, 0]).tolist() == [2, 0, 0, 1, 0]
    pair = Automaton(LatticeSpace(5), LatticeNeighborhood(((0,), (1,))), three, 3)
    assert pair.local_map().tolist() == [0, 1, 2, 1, 2, 0, 2, 0, 1]
