"""Test the `_calibration` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import CalibrationResult, calibrate
from skcomplex.core.experiments._calibration import _CalibrationObjective, _check_priors, _numpyro_prior, _series_of
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.economics.markets import BrockHommesSystem
from skcomplex.systems.social.opinion_dynamics import DeffuantSystem


def _logistic_data(noise: float = 0.01, seed: int = 0):
    """Return logistic data at r = 0.8 and K = 500 with relative Gaussian noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    return {'population.x': (times, truth * (1 + noise * rng.standard_normal(times.size)))}


def test_calibration_recovers_logistic_parameters():
    """Test that calibration recovers r = 0.8 and K = 500 from data with 1% noise and fits with R² above 0.99."""
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 21.0)
    truth = 500 / (1 + 49 * np.exp(-0.8 * times))
    data = {'population.x': (times, truth * (1 + 0.01 * rng.standard_normal(times.size)))}
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
    settings = {'data': {'population.x': ([0.0, 1.0], [1.0, 2.0])}, 'bounds': {'r': (0.1, 2.0)}, **arguments}
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
    return {'pricing.deviation': (truth.times[1:], truth.get_values('pricing.deviation')[1:])}


@pytest.mark.slow
def test_global_fit_is_reproducible_and_beats_the_midpoints():
    """Test that two seeded global fits of Brock-Hommes agree and beat the midpoints of the bounds."""
    data, bounds = _brock_hommes_data(), {'intensity': (0.5, 8.0), 'strength': (1.0, 1.6)}
    first = calibrate(BrockHommesSystem(20), data, bounds, method='global', budget=200, random_state=0)
    second = calibrate(BrockHommesSystem(20), data, bounds, method='global', budget=200, random_state=0)
    assert first.estimates == second.estimates
    objective = _CalibrationObjective(BrockHommesSystem(20), _series_of(data), _check_priors(bounds), random_state=0)
    best = objective.get_loss(list(first.estimates.values()), 'squares')
    assert best < objective.get_loss([4.25, 1.3], 'squares')


def test_event_likelihood_recovers_the_confidence_of_deffuant():
    """Test that the event likelihood of a recorded Deffuant run recovers the confidence within 0.02."""
    recorded = simulate(
        DeffuantSystem(100, confidence=0.25, steepness=50.0),
        until=50,
        random_state=0,
        record_draws=True,
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


LOGISTIC_BOUNDS = {'r': (0.1, 2.0), 'K': (100.0, 1000.0)}


@pytest.mark.parametrize(
    'prior',
    [
        stats.uniform(0.2, 0.5),
        stats.norm(0.8, 0.1),
        stats.lognorm(0.3, scale=0.8),
        stats.truncnorm(-1.0, 2.0, loc=0.8, scale=0.1),
        stats.beta(2.0, 3.0),
        stats.gamma(2.0, scale=0.4),
        stats.expon(scale=0.5),
    ],
    ids=['uniform', 'norm', 'lognorm', 'truncnorm', 'beta', 'gamma', 'expon'],
)
def test_numpyro_priors_match_their_scipy_densities(prior):
    """Test that every supported SciPy family maps to a numpyro distribution of the same density."""
    points = prior.ppf([0.1, 0.5, 0.9])
    mapped = _numpyro_prior('r', prior)
    np.testing.assert_allclose(np.asarray(mapped.log_prob(points)), prior.logpdf(points), rtol=1e-5)


def test_mcmc_rejects_an_unsupported_prior_family():
    """Test that a Cauchy prior raises a named error for MCMC."""
    bounds = {'r': stats.cauchy(0.8, 0.1), 'K': (100.0, 1000.0)}
    with pytest.raises(ValidationError, match='cauchy distribution, which MCMC and variational calibration'):
        calibrate(LogisticGrowthSystem(n=10.0), _logistic_data(), bounds, method='mcmc', loss='likelihood')


@pytest.mark.slow
def test_mcmc_samples_the_logistic_posterior_with_converged_chains():
    """Test that four NUTS chains of 500 samples give a split R-hat below 1.05 and an ESS above 200."""
    result = calibrate(
        LogisticGrowthSystem(n=10.0),
        _logistic_data(0.02),
        LOGISTIC_BOUNDS,
        method='mcmc',
        loss='likelihood',
        n_samples=500,
        chains=4,
        random_state=0,
    )
    assert result.diagnostics['sampler'] == 'nuts'
    assert all(value < 1.05 for value in result.diagnostics['rhat'].values())
    assert all(value > 200 for value in result.diagnostics['ess'].values())
    assert result.samples['r'].size == 2000
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.estimates['r'] == pytest.approx(0.8, rel=0.05)
    assert result.success


@pytest.mark.slow
def test_mcmc_with_a_normal_prior_narrows_it():
    """Test that MCMC under the prior norm(0.8, 0.05) on r gives samples narrower than the prior near 0.8."""
    bounds = {'r': stats.norm(0.8, 0.05), 'K': (100.0, 1000.0)}
    result = calibrate(
        LogisticGrowthSystem(n=10.0),
        _logistic_data(0.02),
        bounds,
        method='mcmc',
        loss='likelihood',
        n_samples=300,
        random_state=0,
    )
    assert np.std(result.samples['r']) < 0.05
    assert abs(np.mean(result.samples['r']) - 0.8) < 0.1


@pytest.mark.slow
def test_mcmc_of_the_event_likelihood_runs_the_ensemble_sampler():
    """Test that MCMC of a recorded Gillespie run uses the gradient-free ensemble sampler and recovers the rates."""
    system = SIRSystem(198, 2, beta=0.0015, gamma=0.1, stochastic=True)
    recorded = simulate(system, until=60, random_state=3, record_draws=True)
    bounds = {'beta': (0.0003, 0.005), 'gamma': (0.02, 0.4)}
    result = calibrate(
        SIRSystem(198, 2, stochastic=True),
        recorded,
        bounds,
        method='mcmc',
        loss='events',
        n_samples=100,
        random_state=0,
    )
    assert result.diagnostics['sampler'] == 'aies'
    assert result.estimates['beta'] == pytest.approx(0.0015, rel=0.25)
    assert result.estimates['gamma'] == pytest.approx(0.1, rel=0.25)


def _weighted_interval(values, weights, level=0.9):
    """Return the central interval of weighted samples."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    tail = (1 - level) / 2
    return values[order][np.searchsorted(cumulative, tail)], values[order][np.searchsorted(cumulative, 1 - tail)]


@pytest.mark.slow
def test_abc_smc_covers_the_rates_of_a_stochastic_epidemic():
    """Test that the 90% intervals of ABC-SMC of the infected counts of one run hold the true rates.

    The transmission rate 0.3 per day of a well-mixed population of 200 is the mass-action rate 0.3 / 200.
    """
    system = SIRSystem(198, 2, beta=0.0015, gamma=0.1, stochastic=True)
    truth = simulate(system, until=60, random_state=3)
    times = np.arange(0.0, 61.0, 3.0)
    data = {'I.x': (times, np.interp(times, truth.times, truth.get_values('I.x')))}
    bounds = {'beta': (0.0003, 0.005), 'gamma': (0.02, 0.4)}
    result = calibrate(
        SIRSystem(198, 2, stochastic=True),
        data,
        bounds,
        method='abc_smc',
        n_samples=150,
        budget=3000,
        random_state=0,
    )
    for name, value in (('beta', 0.0015), ('gamma', 0.1)):
        low, high = _weighted_interval(result.samples[name], result.weights)
        assert low <= value <= high
    epsilons, rates = result.diagnostics['epsilons'], result.diagnostics['acceptance_rates']
    assert epsilons.size == rates.size >= 2
    assert np.all((rates > 0) & (rates <= 1))
    assert result.weights.sum() == pytest.approx(1.0)
    assert set(result.predictive) == {'q05', 'q50', 'q95', 'coverage'}


def test_abc_smc_compares_summary_statistics():
    """Test that ABC-SMC compares the summaries of the series, here the three final counts."""
    truth = simulate(SIRSystem(48, 2, beta=0.006, stochastic=True), until=30, random_state=0)
    times = np.arange(0.0, 31.0, 5.0)
    data = {path: (times, np.interp(times, truth.times, truth.get_values(path))) for path in ('S.x', 'I.x', 'R.x')}

    def finals(series):
        return [series[path][-1] for path in ('S.x', 'I.x', 'R.x')]

    result = calibrate(
        SIRSystem(48, 2, stochastic=True),
        data,
        {'beta': (0.001, 0.02)},
        method='abc_smc',
        summary=finals,
        n_samples=50,
        budget=300,
        random_state=0,
    )
    assert result.diagnostics['summaries'] == 3


def test_variational_calibration_needs_a_differentiable_system():
    """Test that variational calibration of a stochastic system names the tag and the other methods."""
    data = {'I.x': ([0.0, 1.0], [10.0, 12.0])}
    with pytest.raises(ValidationError, match=r'differentiable\. Use one of the methods local, global, abc_smc, mcmc'):
        calibrate(SIRSystem(stochastic=True), data, {'beta': (0.0001, 0.001)}, method='variational')


@pytest.mark.parametrize('loss', ['likelihood', 'squares'])
def test_variational_calibration_recovers_the_logistic_parameters(loss):
    """Test that variational calibration through the JAX run recovers r and K."""
    result = calibrate(
        LogisticGrowthSystem(n=10.0),
        _logistic_data(0.02),
        LOGISTIC_BOUNDS,
        method='variational',
        loss=loss,
        n_samples=300,
        random_state=0,
    )
    assert result.estimates['r'] == pytest.approx(0.8, rel=0.05)
    assert result.estimates['K'] == pytest.approx(500, rel=0.05)
    assert result.diagnostics['losses'].size == 2000
    assert result.success


@pytest.mark.slow
def test_predictive_band_covers_a_well_specified_model():
    """Test that the 90% predictive band of MCMC on logistic data with 5% noise holds 75% to 100% of the data."""
    result = calibrate(
        LogisticGrowthSystem(n=10.0),
        _logistic_data(0.05, seed=1),
        LOGISTIC_BOUNDS,
        method='mcmc',
        loss='likelihood',
        n_samples=300,
        random_state=0,
    )
    predictive = result.predictive
    assert 0.75 <= predictive['coverage'] <= 1
    assert np.all(predictive['q05']['population.x'] <= predictive['q95']['population.x'])
    assert predictive['q50']['population.x'].shape == (21,)
