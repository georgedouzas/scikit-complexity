"""Test the `_calibration` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import CalibrationResult, calibrate
from skcomplex.core.experiments._calibration import _CalibrationObjective, _check_priors, _series_of
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.economics.markets import BrockHommesSystem
from skcomplex.systems.social.opinion_dynamics import DeffuantSystem


def _logistic_data(noise: float = 0.01, seed: int = 0):
    """Return logistic data at r = 0.8 and K = 500 with relative Gaussian noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    return {'population.n': (times, truth * (1 + noise * rng.standard_normal(times.size)))}


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
        ({'method': 'nested'}, 'Use one of local, global, abc_smc, mcmc, variational'),
        ({'method': 'abc_smc', 'loss': 'events'}, "'abc_smc' takes the losses squares"),
        ({'method': 'mcmc', 'loss': 'squares'}, "'mcmc' takes the losses likelihood, events"),
        ({'loss': 'events'}, 'recorded with'),
        ({'bounds': {'r': 'wide'}}, 'a pair'),
        ({'bounds': {'r': (2.0, 0.1)}}, 'low < high'),
        ({'replicates': 0}, 'replicates = 0'),
        ({'data': {}}, 'observed series'),
        ({'bounds': {'growth.r': (0.1, 2.0)}}, r'not the dotted paths growth\.r'),
    ],
    ids=['loss', 'method', 'abc-loss', 'mcmc-loss', 'events-data', 'string', 'order', 'replicates', 'data', 'path'],
)
def test_calibration_rejects_invalid_settings(arguments, message):
    """Test that bad methods, losses, bounds, replicates, data, or dotted paths raise a named error."""
    settings = {'data': {'population.n': ([0.0, 1.0], [1.0, 2.0])}, 'bounds': {'r': (0.1, 2.0)}, **arguments}
    with pytest.raises(ValidationError, match=message):
        calibrate(LogisticGrowthSystem(), **settings)


def test_default_method_is_the_local_fit():
    """Test that a call without a method gives the estimates of `method='local'`."""
    bounds = {'r': (0.1, 2.0), 'K': (100.0, 1000.0)}
    default = calibrate(LogisticGrowthSystem(n=10.0), _logistic_data(), bounds)
    local = calibrate(LogisticGrowthSystem(n=10.0), _logistic_data(), bounds, method='local')
    assert default.estimates == local.estimates


def test_point_fit_holds_no_posterior():
    """Test that a local fit holds `None` in the posterior fields and the Jacobian covariance."""
    result = calibrate(LogisticGrowthSystem(n=10.0), _logistic_data(), {'r': (0.1, 2.0), 'K': (100.0, 1000.0)})
    assert result.samples is None
    assert result.weights is None
    assert result.predictive is None
    assert result.covariance.shape == (2, 2)
    assert np.all(np.diag(result.covariance) > 0)


def test_point_fit_searches_the_central_part_of_a_scipy_prior():
    """Test that a normal prior gives the search box of its central 99.9% and still fits."""
    priors = _check_priors({'r': stats.norm(0.8, 0.1), 'K': (100.0, 1000.0)})
    objective = _CalibrationObjective(LogisticGrowthSystem(n=10.0), _series_of(_logistic_data()), priors)
    low, high = objective.box
    np.testing.assert_allclose(low, [stats.norm(0.8, 0.1).ppf(0.0005), 100.0])
    np.testing.assert_allclose(high, [stats.norm(0.8, 0.1).ppf(0.9995), 1000.0])
    bounds = {'r': stats.norm(0.8, 0.1), 'K': (100.0, 1000.0)}
    result = calibrate(LogisticGrowthSystem(n=10.0), _logistic_data(), bounds)
    assert result.estimates['r'] == pytest.approx(0.8, rel=0.05)


def _brock_hommes_data():
    """Return the price deviations of a small Brock-Hommes market."""
    truth = simulate(BrockHommesSystem(20, intensity=4.0, strength=1.3), until=60, random_state=0)
    return {'pricing.deviation': (truth.times[1:], truth.values('pricing.deviation')[1:])}


def test_global_fit_is_reproducible_and_beats_the_midpoints():
    """Test that two seeded global fits of Brock-Hommes agree and beat the midpoints of the bounds."""
    data, bounds = _brock_hommes_data(), {'intensity': (0.5, 8.0), 'strength': (1.0, 1.6)}
    first = calibrate(BrockHommesSystem(20), data, bounds, method='global', budget=200, random_state=0)
    second = calibrate(BrockHommesSystem(20), data, bounds, method='global', budget=200, random_state=0)
    assert first.estimates == second.estimates
    objective = _CalibrationObjective(BrockHommesSystem(20), _series_of(data), _check_priors(bounds), random_state=0)
    best = objective.loss(list(first.estimates.values()), 'squares')
    assert best < objective.loss([4.25, 1.3], 'squares')


def test_event_likelihood_recovers_the_confidence_of_deffuant():
    """Test that the event likelihood of a recorded Deffuant run recovers the confidence within 0.02."""
    recorded = simulate(
        DeffuantSystem(100, confidence=0.25, steepness=50.0), until=50, random_state=0, record_draws=True,
    )
    result = calibrate(
        DeffuantSystem(100, steepness=50.0),
        recorded,
        {'confidence': (0.05, 0.5)},
        loss='events',
        random_state=0,
    )
    assert result.estimates['confidence'] == pytest.approx(0.25, abs=0.02)
    assert result.covariance is None


def test_event_likelihood_recovers_the_rates_of_a_gillespie_run():
    """Test that the event likelihood of one Gillespie run of 1000 people recovers beta and gamma within 10%."""
    recorded = simulate(SIRSystem(stochastic=True), until=100, random_state=0, record_draws=True)
    bounds = {'beta': (0.0001, 0.001), 'gamma': (0.02, 0.5)}
    result = calibrate(SIRSystem(stochastic=True), recorded, bounds, loss='events', random_state=0)
    assert result.estimates['beta'] == pytest.approx(0.0003, rel=0.1)
    assert result.estimates['gamma'] == pytest.approx(0.1, rel=0.1)
    assert result.r2 == pytest.approx(1.0)
