"""Test the `_banking` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.systems.economics.macroeconomics import FractionalReserveBankingSystem

from ..conftest import _run_observed


def test_fractional_reserve_money_multiplier():
    """Broad money settles at base money over the reserve ratio, and moves to the new multiplier when it changes."""
    trajectory = _run_observed(
        FractionalReserveBankingSystem(ratio=0.1, new_ratio=0.2, change_at=100.0).prepare().system_,
        200,
        observables=FractionalReserveBankingSystem().get_run_settings()['observables'],
    )
    np.testing.assert_allclose(trajectory.values('reserves'), 1000.0)
    assert trajectory.values('broad_money')[90:100].mean() == pytest.approx(10_000.0, rel=0.05)
    assert trajectory.values('broad_money')[190:].mean() == pytest.approx(5_000.0, rel=0.05)
    ratio = trajectory.values('banks.reserve_ratio')
    assert ratio[99] == pytest.approx(0.1)
    assert ratio[-1] == pytest.approx(0.2)
