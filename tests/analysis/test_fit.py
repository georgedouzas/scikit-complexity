"""Test the fits of power laws, stable laws, extreme-value laws, and named laws."""

import warnings

import networkx as nx
import numpy as np
import pytest
from scipy import stats
from sklearn.base import clone

from skcomplex.analysis import FitAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run


def _fit(law: str, source: object, **settings: object) -> FitAnalysis:
    """Return the run fit of a law to a source."""
    return FitAnalysis(law, **settings).prepare(source).run()


def _truncated_power_law(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return samples of `x^-1.5 exp(-x / 100)` above 1 by rejection."""
    samples = np.zeros(0)
    while samples.size < size:
        candidates = (1 - rng.random(4 * size)) ** (-1 / 0.5)
        kept = candidates[rng.random(candidates.size) < np.exp(-(candidates - 1) / 100)]
        samples = np.concatenate([samples, kept])
    return samples[:size]


def _random_exchange_economy(rng: np.random.Generator, agents: int, exchanges: int) -> np.ndarray:
    """Return the money of agents after random pairs pool their money and split it at a uniform fraction."""
    money = np.ones(agents)
    for _ in range(exchanges):
        order = rng.permutation(agents)
        first, second = order[: agents // 2], order[agents // 2 :]
        pooled = money[first] + money[second]
        share = rng.random(first.size)
        money[first], money[second] = share * pooled, (1 - share) * pooled
    return money


def test_power_law_exponent_recovered():
    """Test that 10,000 samples of a power law with exponent 2.5 above 1 give 2.5 within 0.1 and p above 0.1."""
    samples = (1 - np.random.default_rng(1).random(10_000)) ** (-1 / 1.5)
    fit = _fit('power', samples, bootstrap=50, random_state=0)
    assert fit.exponent_ == pytest.approx(2.5, abs=0.1)
    assert fit.p_value_ > 0.1
    ratio, p_value = fit.comparisons_['exponential']
    assert ratio > 0
    assert p_value < 0.01


def test_exponential_sample_is_not_a_power_law():
    """Test that an exponential sample loses the comparison with an exponential tail."""
    samples = np.random.default_rng(2).exponential(5.0, 5000) + 1
    assert _fit('power', samples, minimum=1.0, bootstrap=0).comparisons_['exponential'][0] < 0


def test_power_law_silences_powerlaw_warnings():
    """Test that the fits of `powerlaw` warn nothing, and that the silencing ends with the run."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _fit('power', np.floor((1 - rng.random(3000)) ** (-1 / 1.5)), discrete=True, bootstrap=0)
        _fit('power', rng.exponential(size=2000), bootstrap=0)
        assert caught == []
        warnings.warn('after the fit', UserWarning, stacklevel=1)
    assert [str(warning.message) for warning in caught] == ['after the fit']


def test_discrete_fit_of_preferential_attachment():
    """Test that the degrees of a Barabási-Albert graph have a tail exponent within 0.3 of 3."""
    degrees = np.array([degree for _, degree in nx.barabasi_albert_graph(10_000, 2, seed=0).degree()])
    assert _fit('power', degrees, discrete=True, bootstrap=0).exponent_ == pytest.approx(3, abs=0.3)


def test_power_law_discrete_samples():
    """Test that 10,000 discrete power-law samples above 1 give their exponent with the exact discrete likelihood."""
    samples = stats.zipf(2.5).rvs(10_000, random_state=np.random.default_rng(0))
    assert _fit('power', samples, discrete=True, bootstrap=0).exponent_ == pytest.approx(2.5, abs=0.1)


def test_power_law_truncated_tail():
    """Test that a power law with an exponential cutoff prefers the truncated power law with p below 0.05."""
    samples = _truncated_power_law(10_000, np.random.default_rng(1))
    fit = _fit('power', samples, minimum=1.0, bootstrap=0, compare_with=('truncated_power_law',))
    ratio, p_value = fit.comparisons_['truncated_power_law']
    assert ratio < 0
    assert p_value < 0.05
    assert list(fit.comparisons_) == ['truncated_power_law']


def test_power_law_upper_cutoff():
    """Test that an upper cutoff limits the tail and is reported."""
    samples = (1 - np.random.default_rng(2).random(5000)) ** (-1 / 1.5)
    fit = _fit('power', samples, minimum=1.0, maximum=100.0, bootstrap=0)
    assert fit.maximum_ == 100.0
    assert fit.tail_ == int(np.sum(samples <= 100))


def test_power_law_matches_previous_estimates():
    """Test that the fixed-cutoff fit equals the previous in-house estimate on a fixed seed."""
    samples = (1 - np.random.default_rng(7).random(5000)) ** (-1 / 1.5)
    fit = _fit('power', samples, minimum=1.0, bootstrap=0)
    assert fit.exponent_ == pytest.approx(2.5147885507011036, abs=1e-8)
    assert fit.comparisons_['exponential'][0] == pytest.approx(1619.996186992908, rel=1e-6)


def test_power_law_needs_a_tail():
    """Test that a fixed cutoff above all but a few samples raises an error."""
    with pytest.raises(AnalysisError, match='at least 10'):
        _fit('power', np.arange(1.0, 20.0), minimum=15.0, bootstrap=0)


def test_extreme_value_classes():
    """Test that block maxima of exponential, Pareto, and uniform samples fall in the three extreme-value classes."""
    rng = np.random.default_rng(0)
    size, block = 1_000_000, 1000
    assert _fit('extreme-value', rng.exponential(size=size), block=block).shape_ == pytest.approx(0, abs=0.05)
    assert _fit('extreme-value', (1 - rng.random(size)) ** (-1 / 2), block=block).shape_ == pytest.approx(0.5, abs=0.1)
    assert _fit('extreme-value', rng.random(size), block=block).shape_ == pytest.approx(-1, abs=0.1)
    with pytest.raises(AnalysisError, match='at least 10 blocks'):
        _fit('extreme-value', np.zeros(50), block=10)


def test_stable_law_of_cauchy_sums():
    """Test that normalized sums of Cauchy values fit a stable law of index 1."""
    sums = np.random.default_rng(2).standard_cauchy((2000, 10)).sum(axis=1) / 10
    assert _fit('stable', sums).index_ == pytest.approx(1, abs=0.15)


def test_named_law_boltzmann_law_on_two_domains():
    """Test that gas energies and exchanged money are exponential at the energy per molecule and money per agent."""
    velocities = np.asarray(run('ideal_gas', settings={'random_state': 0}).trajectory_.values('gas.x_d1'), dtype=float)
    energies = 0.5 * np.sum(velocities**2, axis=-1)
    gas = _fit('exponential', energies[-1], fixed={'loc': 0})
    assert gas.p_value_ > 0.05
    assert gas.parameters_['scale'] == pytest.approx(energies[0].mean(), rel=0.02)
    money = _random_exchange_economy(np.random.default_rng(3), 2000, 400)
    economy = _fit('exponential', money, fixed={'loc': 0})
    assert economy.p_value_ > 0.05
    assert economy.parameters_['scale'] == pytest.approx(1, rel=0.02)


def test_maxwell_law_of_the_speeds_of_an_ideal_gas():
    """Test that the speeds of a three-dimensional ideal gas follow the Maxwell law of its kinetic temperature."""
    trajectory = run('ideal_gas', n_dim=3, random_state=0, settings={'until': 1.0}).trajectory_
    velocities = np.asarray(trajectory.values('gas.x_d1'), dtype=float)
    pooled = FitAnalysis('maxwell', group='gas').prepare(trajectory).values_
    np.testing.assert_allclose(np.sort(pooled), np.sort(np.linalg.norm(velocities, axis=-1).ravel()))
    fit = _fit('maxwell', np.linalg.norm(velocities[-1], axis=-1), fixed={'loc': 0})
    assert fit.p_value_ > 0.05
    assert fit.parameters_['scale'] ** 2 == pytest.approx(np.mean(velocities[-1] ** 2), rel=0.05)


def test_unknown_law_lists_the_laws():
    """Test that an unknown law and an unknown alternative raise errors that list the choices."""
    with pytest.raises(AnalysisError, match='power, stable'):
        FitAnalysis('zipf').prepare()
    with pytest.raises(AnalysisError, match='compare_with'):
        FitAnalysis(compare_with=('cauchy',)).prepare()


def test_setting_the_law_ignores_is_rejected():
    """Test that a changed setting the law does not read raises an error naming it."""
    with pytest.raises(AnalysisError, match='does not read block'):
        FitAnalysis('power', block=10).prepare()
    with pytest.raises(AnalysisError, match='does not read minimum'):
        FitAnalysis('stable', minimum=1.0).prepare()


def test_missing_settings_are_rejected():
    """Test that an extreme-value law without blocks raises an error."""
    with pytest.raises(AnalysisError, match='block'):
        FitAnalysis('extreme-value').prepare()


def test_clone_keeps_settings_and_drops_results():
    """Test that a clone of a run fit has its settings and none of its results."""
    fit = _fit('power', (1 - np.random.default_rng(0).random(500)) ** (-1 / 1.5), bootstrap=0)
    copy = clone(fit)
    assert copy.get_params() == fit.get_params()
    assert not hasattr(copy, 'exponent_')
