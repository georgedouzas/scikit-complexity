"""Test the `_calibration` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.experiments import (
    calibrate,
)


def test_calibration_recovers_logistic_parameters():
    """Test that calibration recovers r = 0.8 and K = 500 from data with 1% noise and fits with R² above 0.99."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    data = {'population.n': (times, truth * (1 + 0.01 * rng.standard_normal(times.size)))}
    result = calibrate('logistic_growth', data, {'growth.r': (0.1, 2.0), 'growth.K': (100.0, 1000.0)})
    assert result.estimates['growth.r'] == pytest.approx(0.8, rel=0.05)
    assert result.estimates['growth.K'] == pytest.approx(500, rel=0.05)
    assert result.r2 > 0.99
    assert result.covariance is not None
