"""Test histograms, survival functions, moments, moment ratios, dispersion, inequality, tails, records, distances."""

import numpy as np
import pytest

from skcomplex.core.analysis import DistributionResult, distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError, ValidationError
from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem, IsingSystem


def _distribution(kind: str, source: object, **settings: object) -> DistributionResult:
    """Return the run distribution analysis of a source for one kind."""
    return distribution(source, kind=kind, **settings)


def test_gini_and_tail_index():
    """Test the Gini coefficient of equal and concentrated values and the Hill tail index."""
    assert _distribution('gini', [5, 5, 5, 5]).gini == 0
    assert _distribution('gini', [0, 0, 0, 1]).gini == pytest.approx(0.75)
    samples = (1 - np.random.default_rng(3).random(50_000)) ** (-1 / 1.5)
    assert _distribution('tail-index', samples, tail=5000).tail_index == pytest.approx(1.5, abs=0.1)
    with pytest.raises(AnalysisError, match='tail = 50000'):
        _distribution('tail-index', samples, tail=50_000)


def test_histogram_and_ccdf():
    """Test that log-binned densities integrate to 1 and the survival function starts at 1."""
    samples = (1 - np.random.default_rng(4).random(1000)) ** (-1 / 1.5)
    histogram = _distribution('histogram', samples, log=True)
    assert np.sum(histogram.density * np.diff(histogram.edges)) == pytest.approx(1.0)
    survival = _distribution('ccdf', samples).survival
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) < 0)


def test_logarithmic_bins_need_positive_values():
    """Test that logarithmic bins of a sample with zeros raise an error."""
    with pytest.raises(AnalysisError, match='positive values'):
        _distribution('histogram', [0.0, 1.0, 2.0], log=True)


def test_complex_values_are_rejected():
    """Test that complex values raise an error that names the real views."""
    with pytest.raises(ValidationError, match='real part'):
        _distribution('ccdf', np.array([1 + 1j, 2.0]))


def test_record_counts_of_independent_values():
    """Test that 10,000 sequences of 1000 independent values hold the harmonic number of records on average."""
    counts = _distribution('records', np.random.default_rng(1).random((10_000, 1000)).T).records
    harmonic = np.sum(1 / np.arange(1, 1001))
    assert counts.mean() == pytest.approx(harmonic, rel=0.01)
    assert _distribution('records', np.array([[1, 4], [3, 3], [2, 2], [5, 1]])).records.tolist() == [3, 1]


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind lists the kinds and a setting the kind ignores is rejected."""
    with pytest.raises(AnalysisError, match='histogram, ccdf'):
        distribution(np.zeros(10), kind='mean')
    with pytest.raises(AnalysisError, match='does not read bins'):
        distribution(np.zeros(10), kind='gini', bins=10)
    with pytest.raises(AnalysisError, match='two positive even orders'):
        distribution(np.zeros(10), kind='moment-ratio', orders=(3, 2))


def _bursty_expression(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Return mRNA copy numbers sampled once per unit time from bursts of geometric size with mean 4 and decay 1."""
    burst_rate, decay, mean_burst = 5.0, 1.0, 4.0
    recorded: list[int] = []
    copies, time, next_sample = 0, 0.0, 50.0
    while len(recorded) < samples:
        total = burst_rate + decay * copies
        step = rng.exponential(1 / total)
        while next_sample < time + step and len(recorded) < samples:
            recorded.append(copies)
            next_sample += 1.0
        time += step
        if rng.random() < burst_rate / total:
            copies += rng.geometric(1 / (1 + mean_burst)) - 1
        else:
            copies -= 1
    return np.asarray(recorded, dtype=float)


def test_moment_ratio_of_any_sample():
    """Test the moment ratio of the magnetisation samples of an Ising system and of the returns of a market system."""
    ordered = simulate(
        IsingSystem(size=16, temperature=1.5, initial='up').prepare().system_,
        until=300,
        dt=1,
        random_state=0,
    )
    hot = simulate(IsingSystem(size=16, temperature=20.0).prepare().system_, until=300, dt=1, random_state=0)
    magnetisations = np.column_stack(
        [trajectory.get_lattice('spins.spin')[50:].mean(axis=(1, 2)) for trajectory in (ordered, hot)],
    )
    ising = _distribution('moment-ratio', magnetisations)
    assert ising.cumulant[0] == pytest.approx(2 / 3, abs=0.02)
    assert abs(ising.cumulant[1]) < 0.15
    market = simulate(GeometricBrownianMotionSystem(), until=100, random_state=0, dt=0.01)
    returns = np.diff(np.log(market.get_values('asset.price')))
    prices = _distribution('moment-ratio', returns)
    assert prices.ratio[0] == pytest.approx(3, rel=0.1)
    assert prices.cumulant[0] == pytest.approx(1 - prices.ratio[0] / 3)
    assert abs(prices.cumulant[0]) < 0.05
    sixth = _distribution('moment-ratio', np.random.default_rng(0).standard_normal(100_000), orders=(6, 2))
    assert sixth.ratio[0] == pytest.approx(15, rel=0.05)


def test_moments_of_absolute_samples_and_of_agent_speeds():
    """Test the moments of signed and absolute samples, and of the speeds of an ideal gas at every frame."""
    moments = _distribution('moments', np.array([[1.0, -2.0], [-3.0, 4.0]]), absolute=True)
    assert moments.mean.tolist() == [2.0, 3.0]
    assert moments.variance.tolist() == [1.0, 1.0]
    gaussian = _distribution('moments', np.random.default_rng(0).standard_normal(100_000))
    assert gaussian.skewness[0] == pytest.approx(0, abs=0.03)
    assert gaussian.kurtosis[0] == pytest.approx(0, abs=0.05)
    trajectory = simulate(IdealGasSystem(n_dim=3, random_state=0), until=1.0, random_state=0)
    speeds = _distribution('moments', trajectory)
    velocities = np.asarray(trajectory.get_values('gas.x_d1'), dtype=float)
    assert speeds.mean.shape == (len(trajectory.times),)
    np.testing.assert_allclose(speeds.mean, np.linalg.norm(velocities, axis=-1).mean(axis=1))
    np.testing.assert_allclose(_distribution('moments', trajectory, group='gas').mean, speeds.mean)


def test_dispersion_index_of_poisson_and_bursty_production():
    """Test that Gillespie production counts give a dispersion index near 1 and bursty mRNA copies near 5."""
    trajectory = simulate(PoissonProcessSystem(n_processes=1), until=50_000.0, random_state=0)
    counts = np.diff(trajectory.get_values('counters.count')[:, 0][::10])
    assert _distribution('dispersion-index', counts).index[0] == pytest.approx(1, abs=0.05)
    bursty = _bursty_expression(np.random.default_rng(0), 20_000)
    assert _distribution('dispersion-index', bursty).index[0] == pytest.approx(5, abs=0.3)
    with pytest.raises(AnalysisError, match='mean 0'):
        _distribution('dispersion-index', [0, 0])


def test_total_variation_of_separated_and_mixed_groups():
    """Test that separated groups give 1, even mixing 0, and more than two groups or no units an error."""
    assert _distribution('total-variation', np.column_stack([[0, 0, 1, 1], [0, 0, 1, 1]])).distance == 1
    assert _distribution('total-variation', np.column_stack([[0, 1, 0, 1], [0, 0, 1, 1]])).distance == 0
    with pytest.raises(AnalysisError, match='two groups'):
        _distribution('total-variation', np.column_stack([[0, 1, 2], [0, 0, 1]]))
    with pytest.raises(AnalysisError, match='one row per item'):
        _distribution('total-variation', [0, 1])
