"""Test the `_calibration` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import CalibrationExperiment
from skcomplex.systems.biology.ecology import LogisticGrowthSystem


def test_calibration_recovers_logistic_parameters():
    """Test that calibration recovers r = 0.8 and K = 500 from data with 1% noise and fits with R² above 0.99."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    data = {'population.n': (times, truth * (1 + 0.01 * rng.standard_normal(times.size)))}
    bounds = {'growth.r': (0.1, 2.0), 'growth.K': (100.0, 1000.0)}
    calibration = CalibrationExperiment(LogisticGrowthSystem, data, bounds).prepare().run()
    assert calibration.estimates_['growth.r'] == pytest.approx(0.8, rel=0.05)
    assert calibration.estimates_['growth.K'] == pytest.approx(500, rel=0.05)
    assert calibration.r2_ > 0.99
    assert calibration.covariance_ is not None
    assert 0 < calibration.mape_ < 5
    assert calibration.rmse_ >= calibration.mae_ > 0


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [({'loss': 'absolute'}, 'Unknown loss'), ({'replicates': 0}, 'replicates = 0'), ({'data': {}}, 'observed series')],
    ids=['loss', 'replicates', 'data'],
)
def test_calibration_check_rejects_invalid_settings(arguments, message):
    """Test that an unknown loss, zero replicates, or no data build and fail the check with a named error."""
    settings = {'data': {'population.n': ([0.0, 1.0], [1.0, 2.0])}, **arguments}
    with pytest.raises(ValidationError, match=message):
        CalibrationExperiment(LogisticGrowthSystem, bounds={'growth.r': (0.1, 2.0)}, **settings).prepare()
