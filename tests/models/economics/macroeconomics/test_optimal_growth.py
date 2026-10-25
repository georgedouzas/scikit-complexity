"""Test the `_optimal_growth` module of `skcomplex.models.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.models import run


def test_optimal_consumption_approaches_the_golden_rule():
    """Test that capital rises towards the modified golden rule `alpha k^(alpha - 1) = delta + rho`."""
    trajectory = run('optimal_consumption').trajectory_
    golden = (0.3 / (0.05 + 0.03)) ** (1 / 0.7)
    capital, consumption = trajectory.values('economy.k'), trajectory.values('economy.c')
    early = trajectory.times < 30.0
    assert np.all(np.diff(capital[early]) > 0)
    assert np.all(np.diff(consumption[early]) > 0)
    assert capital[-1] == pytest.approx(golden, rel=0.1)
    assert trajectory.values('saving_rate')[0] > trajectory.values('saving_rate')[-1]
