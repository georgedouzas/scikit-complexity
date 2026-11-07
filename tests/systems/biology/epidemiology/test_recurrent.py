"""Test the `_recurrent` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np
import pytest

from skcomplex.systems.biology.epidemiology import DelayedSIRSSystem

from ..conftest import _run_until


def test_delayed_sirs_without_immunity_is_the_logistic_map():
    """Test that the discrete SIRS system without immunity and visits reduces to the logistic map."""
    trajectory = _run_until(DelayedSIRSSystem(0.1, 0.0, a=2.5, recovery=0, e=0.0), 60, dt=1)
    assert trajectory.values('first.x')[-1] == pytest.approx(1 - 1 / 2.5, rel=1e-6)
    assert trajectory.values('second.x')[-1] == 0


def test_delayed_sirs_outbreaks_recur_and_spread():
    """Test that outbreaks recur in the first city and spread to the second through visits."""
    trajectory = _run_until(DelayedSIRSSystem(), 200, dt=1)
    first = trajectory.values('first.x')
    assert trajectory.values('second.x').max() > 0.1
    assert np.sum((first[1:-1] > first[:-2]) & (first[1:-1] > first[2:]) & (first[1:-1] > 0.1)) >= 3
