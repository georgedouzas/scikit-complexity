"""Test the `_stochastic_growth` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.systems import describe
from skcomplex.systems.economics.macroeconomics import BrockMirmanGrowthSystem

from ..conftest import _run


def test_brock_mirman_growth_saves_a_constant_share_of_output():
    """Test the steady state `(alpha beta)^(1 / (1 - alpha))` and the consumption rule `C = (1 - alpha beta) Y`."""
    system = BrockMirmanGrowthSystem(z_low=1.0, z_high=1.0).prepare().system_
    trajectory = _run(system, 60, observables=describe('brock_mirman_growth').observables_of(system))
    alpha, beta = 1 / 3, 0.95
    assert trajectory.values('economy.K')[-1] == pytest.approx((alpha * beta) ** (1 / (1 - alpha)))
    ratio = trajectory.values('economy.consumption') / trajectory.values('economy.output')
    assert ratio == pytest.approx(np.full(len(ratio), 1 - alpha * beta))


def test_brock_mirman_growth_log_capital_fluctuates_around_its_stationary_mean():
    """Test that `ln K` follows `alpha beta z K^alpha` and averages `(ln(alpha beta) + E ln z) / (1 - alpha)`."""
    system = BrockMirmanGrowthSystem().prepare().system_
    trajectory = _run(system, 4000, observables=describe('brock_mirman_growth').observables_of(system))
    capital, productivity = trajectory.values('economy.K'), trajectory.values('economy.productivity')
    alpha, beta = 1 / 3, 0.95
    assert capital[1:] == pytest.approx(alpha * beta * productivity[:-1] * capital[:-1] ** alpha)
    assert set(np.round(productivity, 6)) == {0.9, 1.1}
    expected = (np.log(alpha * beta) + 0.5 * np.log(0.9 * 1.1)) / (1 - alpha)
    assert np.log(capital[100:]).mean() == pytest.approx(expected, abs=0.02)
