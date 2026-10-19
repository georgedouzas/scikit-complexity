"""Test difference equations of any order through states that keep past values."""

import numpy as np
import pytest

from skcomplex.core import ValidationError, state
from skcomplex.model import Element, System, Update, lag
from skcomplex.runtime import Simulation

Fibonacci = Element.define(
    'Fibonacci',
    states={'x': state(default=1.0, lags=1, initial=(1.0,))},
    laws=lambda s: [Update(s.x, s.x + lag(s.x, 1))],
)
Income = Element.define(
    'Income',
    states={'Y': state(default=110.0, lags=1, initial=(100.0,))},
    parameters={'c': 0.8, 'v': 1.5, 'G': 20.0},
    laws=lambda s: [Update(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)],
)
TooFar = Element.define(
    'TooFar',
    states={'x': state(default=1.0, lags=1)},
    laws=lambda s: [Update(s.x, lag(s.x, 2))],
)


def test_fibonacci_recurrence():
    """Test that `x_{t+1} = x_t + x_{t-1}` from 1 and 1 gives the Fibonacci sequence."""
    trajectory = Simulation(System(elements=[Fibonacci('f')])).run(until=8).trajectory_
    assert trajectory['f.x'].tolist() == [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert trajectory['f.x_lag1'].tolist()[:3] == [1, 1, 2]


def test_multiplier_accelerator_oscillates_with_growing_amplitude():
    """Test that Samuelson's model with c v > 1 oscillates around G / (1 - c) with growing amplitude."""
    income = Simulation(System(elements=[Income('economy')])).run(until=40).trajectory_['economy.Y']
    deviation = income - 20.0 / (1 - 0.8)
    assert np.any(deviation[1:] * deviation[:-1] < 0)
    first, last = np.abs(deviation[:10]).max(), np.abs(deviation[-10:]).max()
    assert last > 10 * first
    roots = np.roots([1, -0.8 * 2.5, 0.8 * 1.5])
    assert np.all(np.abs(roots) > 1)


def test_lag_beyond_kept_values_is_rejected():
    """Test that reading a lag the state does not keep raises an error naming the lag."""
    with pytest.raises(ValidationError, match=r'lag\(.*, 2\)'):
        Simulation(System(elements=[TooFar('t')])).run(until=1)
