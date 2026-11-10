"""Test the `_measurement_errors` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.systems.economics.resources import FisheryMeasurementErrorsSystem

from ..conftest import _run


def test_fishery_measurement_errors_settles_where_the_quota_meets_the_growth():
    """Test that without noise the quota `x - K / 2` holds the stock at `K / sqrt(2)`, below the harvest peak."""
    trajectory = _run(FisheryMeasurementErrorsSystem(2, sigma_r=0.0), 2000)
    assert trajectory.get_values('fisheries.x')[-1] == pytest.approx(np.full(2, 2**-0.5), abs=1e-6)
    assert trajectory.get_values('fisheries.catch')[-1] == pytest.approx(np.full(2, 2**-0.5 - 0.5), abs=1e-6)
    assert float(trajectory.get_values('fisheries.catch')[-1][0]) < 0.25


def test_fishery_measurement_errors_extinction_grows_with_the_noise():
    """Test that stocks never die without noise and that a large measurement error drives many extinct."""

    def extinct(**settings):
        system = FisheryMeasurementErrorsSystem(100, **settings).prepare().system_
        observables = FisheryMeasurementErrorsSystem().get_run_settings()['observables']
        return float(_run(system, 2000, observables=observables).get_values('extinct')[-1])

    assert extinct(sigma_r=0.0) == 0.0
    assert extinct(sigma_r=1.4, error=0.3) > 0.3
    assert extinct(sigma_r=1.4, error=0.3, restocking=True) == 0.0
