"""Test the `_turing_machine` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import TuringMachineSystem


def test_turing_machine_busy_beaver_halts_after_13_steps_with_six_ones():
    """Test that the three-state busy beaver writes six ones and halts after 13 steps, then the tape stays fixed."""
    trajectory = simulate(TuringMachineSystem(), random_state=0)
    symbols, heads = trajectory.get_lattice('tape.symbol'), trajectory.get_lattice('tape.head')
    assert np.all((heads >= 0).sum(axis=1) == 1)
    halted = heads.max(axis=1) == 3
    assert np.argmax(halted) == 13
    assert symbols[13].sum() == 6
    assert np.array_equal(symbols[13], symbols[-1])


def test_turing_machine_follows_a_given_table():
    """Test that a one-state machine that writes ones while moving right fills the tape to its right."""
    table = {('A', 0): (1, 'R', 'A'), ('A', 1): (1, 'R', 'A')}
    trajectory = simulate(TuringMachineSystem(size=11, table=table), until=4, random_state=0)
    assert trajectory.get_lattice('tape.symbol')[-1].tolist() == [0] * 5 + [1] * 4 + [0] * 2


def test_turing_machine_rejects_an_unknown_start():
    """Test that a start state outside the table raises a parameter error."""
    with pytest.raises(ParameterError, match='start'):
        TuringMachineSystem(start='Z').prepare()
