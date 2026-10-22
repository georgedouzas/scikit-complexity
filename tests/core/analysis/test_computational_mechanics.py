"""Test the reconstruction of ε-machines against processes with known causal states."""

import numpy as np
import pytest

from skcomplex.core.analysis import epsilon_machine
from skcomplex.core.base import AnalysisError


def _golden_mean(steps: int, seed: int) -> np.ndarray:
    """Return the golden mean process, fair bits after a 1 and always a 1 after a 0."""
    rng = np.random.default_rng(seed)
    symbols = np.ones(steps, dtype=np.int64)
    draws = rng.random(steps)
    for t in range(1, steps):
        symbols[t] = 1 if symbols[t - 1] == 0 else int(draws[t] < 0.5)
    return symbols


def test_epsilon_machine_golden_mean_process():
    """Test that the golden mean process has two causal states and a statistical complexity of 0.918 bits."""
    machine = epsilon_machine(_golden_mean(100_000, 0), 4)
    assert len(machine.histories) == 2
    assert machine.statistical_complexity == pytest.approx(0.918, abs=0.01)
    assert machine.entropy_rate == pytest.approx(2 / 3, abs=0.01)
    after_zero = next(number for number, state in enumerate(machine.histories) if state[0][-1] == 0)
    assert machine.probabilities[after_zero].tolist() == [0.0, 1.0]
    assert machine.transitions[after_zero, 0] == -1


def test_epsilon_machine_fair_coin_has_one_state():
    """Test that fair coin flips have one causal state, no statistical complexity, and one bit of entropy rate."""
    machine = epsilon_machine(np.random.default_rng(1).integers(0, 2, 50_000), 3)
    assert len(machine.histories) == 1
    assert machine.statistical_complexity == pytest.approx(0.0)
    assert machine.entropy_rate == pytest.approx(1.0, abs=0.01)
    assert machine.transitions.tolist() == [[0, 0]]


def test_epsilon_machine_period_three():
    """Test that a period-three sequence of letters has three states visited in a cycle."""
    machine = epsilon_machine(list('aab') * 500, 3)
    assert machine.alphabet.tolist() == ['a', 'b']
    assert len(machine.histories) == 3
    assert machine.statistical_complexity == pytest.approx(np.log2(3), abs=1e-3)
    assert machine.entropy_rate == pytest.approx(0.0)


def test_epsilon_machine_rejects_long_histories():
    """Test that a history as long as the series raises an error."""
    with pytest.raises(AnalysisError, match='longest history'):
        epsilon_machine([0, 1, 0], 3)
