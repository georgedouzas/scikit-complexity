"""Test the `_banking` module of `skcomplex.models.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.models import describe
from skcomplex.models.economics.macroeconomics import fractional_reserve_banking

from ..conftest import _run_observed


def test_fractional_reserve_money_multiplier():
    """Broad money settles at base money over the reserve ratio, and moves to the new multiplier when it changes."""
    trajectory = _run_observed(
        built := fractional_reserve_banking(ratio=0.1, new_ratio=0.2, change_at=100.0),
        200,
        observables=describe('fractional_reserve_banking').observables_of(built),
    )
    np.testing.assert_allclose(trajectory['reserves'], 1000.0)
    assert trajectory['broad_money'][90:100].mean() == pytest.approx(10_000.0, rel=0.05)
    assert trajectory['broad_money'][190:].mean() == pytest.approx(5_000.0, rel=0.05)
    ratio = trajectory['banks.reserve_ratio']
    assert ratio[99] == pytest.approx(0.1)
    assert ratio[-1] == pytest.approx(0.2)
