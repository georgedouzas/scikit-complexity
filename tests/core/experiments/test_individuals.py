"""Test the `_individuals` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import IndividualCalibrationResult, calibrate_individuals
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.oncology import ImmunoOncologyQspSystem

TIMES = np.arange(0.0, 21.0)
DAYS = np.arange(0.0, 181.0, 7.0)
START = 60.0
QSP_BOUNDS = {'growth': (0.005, 0.1), 'kill': (1e-11, 5e-10)}


def _get_logistic_data(rate):
    """Return the logistic counts of a growth rate from 10 to the capacity 100."""
    return {'population.x': (TIMES, 100 / (1 + 9 * np.exp(-rate * TIMES)))}


def test_calibrate_individuals_fits_every_individual():
    """Test that every individual gets its own fit, in a table with one row per individual."""
    data = {'a': _get_logistic_data(0.4), 'b': _get_logistic_data(0.8)}
    result = calibrate_individuals(LogisticGrowthSystem(), data, {'r': (0.1, 2.0)})
    assert isinstance(result, IndividualCalibrationResult)
    assert list(result.estimates.index) == ['a', 'b']
    assert result.estimates.index.name == 'individual'
    np.testing.assert_allclose(result.estimates['r'], [0.4, 0.8], rtol=1e-3)
    assert result.calibrations['b'].system.r == pytest.approx(0.8, rel=1e-3)


def test_calibrate_individuals_passes_known_parameters_to_a_factory():
    """Test that a factory of systems receives the known parameters of every individual."""
    data = {'a': {'population.x': (TIMES, 100 / (1 + 4 * np.exp(-0.5 * TIMES)))}}
    result = calibrate_individuals(LogisticGrowthSystem, data, {'r': (0.1, 2.0)}, known={'a': {'n': 20.0}})
    assert result.estimates.loc['a', 'r'] == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize(
    ('known', 'match'),
    [({'a': {'r': 0.5}}, "parameter r of the individual 'a' is both known and fitted"), ({'c': {'n': 1.0}}, 'no data')],
)
def test_calibrate_individuals_rejects_bad_known_parameters(known, match):
    """Test that a known parameter that is also fitted, or of an individual without data, raises an error."""
    with pytest.raises(ValidationError, match=match):
        calibrate_individuals(LogisticGrowthSystem(), {'a': _get_logistic_data(0.4)}, {'r': (0.1, 2.0)}, known=known)


def test_calibrate_individuals_needs_an_individual():
    """Test that no data raises an error."""
    with pytest.raises(ValidationError, match='at least one individual'):
        calibrate_individuals(LogisticGrowthSystem(), {}, {'r': (0.1, 2.0)})


@pytest.mark.slow
def test_digital_twins_of_virtual_patients_recover_their_rates():
    """Test that the twins of five patients with their own doses recover growth and killing within 10%.

    The doses start after 60 days without treatment, which separate the growth from the killing.
    """
    rng = np.random.default_rng(0)
    truths, data, known = {}, {}, {}
    for index, dose in enumerate((80.0, 160.0, 240.0, 320.0, 480.0)):
        truth = {'growth': float(rng.uniform(0.01, 0.05)), 'kill': float(rng.uniform(5e-11, 2e-10))}
        patient = ImmunoOncologyQspSystem(dose=dose, start=START, **truth)
        diameters = simulate(patient, until=180.0, dt=1.0).get_values('diameter')
        noisy = diameters[DAYS.astype(int)] * (1 + 0.02 * rng.standard_normal(DAYS.size))
        name = f'patient-{index}'
        truths[name], data[name], known[name] = truth, {'diameter': (DAYS, noisy)}, {'dose': dose}
    result = calibrate_individuals(ImmunoOncologyQspSystem(start=START), data, QSP_BOUNDS, known=known, random_state=0)
    assert len(result.estimates) == 5
    for name, truth in truths.items():
        for parameter, value in truth.items():
            assert result.estimates.loc[name, parameter] == pytest.approx(value, rel=0.1)
