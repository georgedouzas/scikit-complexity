"""Test the `_automaton` module of `skcomplex.models.computer_science.automata`."""

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
