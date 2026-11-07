"""Test the `_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.computer_science.automata import langtons_ant, turing_machine


def test_langtons_ant_moves_and_flips():
    """Test that Langton's ant keeps one ant, flips one cell per step, and builds its highway after 10,000 steps."""
    trajectory = Simulation(langtons_ant((100, 100)), dt=100).prepare().run(until=11_000).trajectory_
    ants = trajectory.lattice('cells.ant')
    assert np.all((ants >= 0).sum(axis=(1, 2)) == 1)
    black = trajectory.lattice('cells.color').sum(axis=(1, 2))
    assert black[1] <= 100
    position = np.argwhere(ants[-1] >= 0)[0]
    assert np.abs(position - 50).max() > 15


def test_turing_machine_busy_beaver_halts_after_13_steps_with_six_ones():
    """Test that the three-state busy beaver writes six ones and halts after 13 steps, then the tape stays fixed."""
    trajectory = create_simulation('turing_machine').prepare().run().trajectory_
    symbols, heads = trajectory.lattice('tape.symbol'), trajectory.lattice('tape.head')
    assert np.all((heads >= 0).sum(axis=1) == 1)
    halted = heads.max(axis=1) == 3
    assert np.argmax(halted) == 13
    assert symbols[13].sum() == 6
    assert np.array_equal(symbols[13], symbols[-1])


def test_turing_machine_follows_a_given_table():
    """Test that a one-state machine that writes ones while moving right fills the tape to its right."""
    table = {('A', 0): (1, 'R', 'A'), ('A', 1): (1, 'R', 'A')}
    trajectory = create_simulation('turing_machine', size=11, table=table).prepare().run(until=4).trajectory_
    assert trajectory.lattice('tape.symbol')[-1].tolist() == [0] * 5 + [1] * 4 + [0] * 2


def test_turing_machine_rejects_an_unknown_start():
    """Test that a start state outside the table raises a parameter error."""
    with pytest.raises(ParameterError, match='start'):
        turing_machine(start='Z')
