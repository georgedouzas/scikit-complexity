"""Test the `_calibration` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import CalibrationResult, calibrate
from skcomplex.systems.biology.ecology import LogisticGrowthSystem


def test_calibration_recovers_logistic_parameters():
    """Test that calibration recovers r = 0.8 and K = 500 from data with 1% noise and fits with R² above 0.99."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    data = {'population.n': (times, truth * (1 + 0.01 * rng.standard_normal(times.size)))}
    bounds = {'r': (0.1, 2.0), 'K': (100.0, 1000.0)}
    system = LogisticGrowthSystem(n=10.0)
    result = calibrate(system, data, bounds)
    assert isinstance(result, CalibrationResult)
    assert result.estimates['r'] == pytest.approx(0.8, rel=0.05)
    assert result.estimates['K'] == pytest.approx(500, rel=0.05)
    assert isinstance(result.system, LogisticGrowthSystem)
    assert result.system.get_params() == {**system.get_params(), **result.estimates}
    assert system.r == 1.0
    assert result.r2 > 0.99
    assert result.covariance is not None
    assert 0 < result.mape < 5
    assert result.rmse >= result.mae > 0


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'loss': 'absolute'}, 'Unknown loss'),
        ({'replicates': 0}, 'replicates = 0'),
        ({'data': {}}, 'observed series'),
        ({'bounds': {'growth.r': (0.1, 2.0)}}, r'not the dotted paths growth\.r'),
    ],
    ids=['loss', 'replicates', 'data', 'path'],
)
def test_calibration_rejects_invalid_settings(arguments, message):
    """Test that an unknown loss, zero replicates, no data, or a dotted path raise a named error."""
    settings = {'data': {'population.n': ([0.0, 1.0], [1.0, 2.0])}, 'bounds': {'r': (0.1, 2.0)}, **arguments}
    with pytest.raises(ValidationError, match=message):
        calibrate(LogisticGrowthSystem(), **settings)
