"""Test the correlation analysis of series, ensembles at two times, fields in space, and overlaps."""

import math

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.analysis import CorrelationAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError
from skcomplex.models import create

RUNS = 20_000
STEP = 0.1


def _correlation(kind: str, source: object, **settings: object) -> CorrelationAnalysis:
    """Return the correlation analysis of a source for one kind, run."""
    return CorrelationAnalysis(kind, **settings).check().prepare(source).run()


def _autoregression(coefficient: float, size: int, seed: int) -> np.ndarray:
    """Return an autoregression of order 1 with unit noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros(size)
    for t in range(1, x.size):
        x[t] = coefficient * x[t - 1] + rng.standard_normal()
    return x


def _ornstein_uhlenbeck(rng: np.random.Generator, frames: int, field: float, switch: int) -> np.ndarray:
    """Return an equilibrium ensemble of `dx = (h - x) dt + sqrt(2) dW` updated exactly, the field on from a frame."""
    decay = np.exp(-STEP)
    spread = np.sqrt(1 - decay**2)
    states = rng.standard_normal(RUNS)
    frames_list = [states.copy()]
    noise_rng = np.random.default_rng(99)
    for frame in range(1, frames):
        forcing = field if frame > switch else 0.0
        states = states * decay + forcing * (1 - decay) + spread * noise_rng.standard_normal(RUNS)
        frames_list.append(states.copy())
    return np.asarray(frames_list).T


def _exact_correlation_length(temperature: float) -> float:
    """Return the exact correlation length of the square Ising lattice above its critical temperature."""
    coupling = 1 / temperature
    return 1 / (-math.log(math.tanh(coupling)) - 2 * coupling)


def _reaction_diffusion_frames(rng: np.random.Generator, frames: int, side: int) -> np.ndarray:
    """Return exact stationary samples of `u_t = 4 lap u - u + noise` on a periodic lattice."""
    lattice = 2 * np.pi * np.fft.fftfreq(side)
    hat = 4 * np.sin(lattice[:, None] / 2) ** 2 + 4 * np.sin(lattice[None, :] / 2) ** 2
    amplitude = 1 / np.sqrt(4 * hat + 1)
    noise = rng.standard_normal((frames, side, side))
    return np.fft.ifft2(np.fft.fft2(noise) * amplitude).real


def test_autocorrelation_of_an_autoregression():
    """Test that an AR(1) series has autocorrelation near its coefficient."""
    series = _autoregression(0.8, 20_000, 7)
    assert _correlation('auto', series, max_lag=1).correlation_[0][1] == pytest.approx(0.8, abs=0.02)


def test_autocorrelation_time_of_an_autoregression():
    """Test that an autoregression with coefficient 0.9 has an autocorrelation time near 9.5."""
    series = _autoregression(0.9, 100_000, 8)
    assert _correlation('auto', series).time_[0] == pytest.approx(9.5, rel=0.15)


def test_autocorrelation_of_every_series():
    """Test that every column gets its autocorrelation and its time, with lags in time."""
    analysis = _correlation('auto', np.column_stack([[1.0, -1.0] * 4, [1.0, 1.0, -1.0, -1.0] * 2]), max_lag=2, dt=0.5)
    assert analysis.correlation_.shape == (2, 3)
    assert analysis.time_.shape == (2,)
    assert analysis.lags_.tolist() == [0.0, 0.5, 1.0]


def test_two_time_correlation_equilibrium_has_no_aging():
    """Test that an equilibrium Ornstein-Uhlenbeck ensemble decays as exp(-tau) at both waiting times."""
    rng = np.random.default_rng(0)
    lags = np.arange(31)
    unperturbed = _ornstein_uhlenbeck(rng, 1031, 0.0, 1000)
    correlation = _correlation('two-time', unperturbed, waiting=[100, 1000], lags=lags).correlation_
    np.testing.assert_allclose(correlation, np.exp(-lags * STEP)[None, :].repeat(2, axis=0), atol=0.03)


def test_two_time_correlation_rejects_late_frames():
    """Test that a waiting time plus a lag beyond the last frame raises an error."""
    with pytest.raises(AnalysisError, match='beyond'):
        _correlation('two-time', np.zeros((2, 5)), waiting=[3], lags=[2])


def test_correlation_function_of_smoothed_noise():
    """Test that smoothing noise over neighbours makes nearby cells correlated and distant ones not."""
    noise = np.random.default_rng(1).normal(size=(128, 128))
    smooth = sum(np.roll(np.roll(noise, dx, 0), dy, 1) for dx in range(-2, 3) for dy in range(-2, 3))
    values = _correlation('function', smooth, max_distance=10).correlation_
    assert values[1] > 0.6
    assert abs(values[10]) < 0.1
    with pytest.raises(AnalysisError, match='constant'):
        _correlation('function', np.ones((4, 4)))


def test_correlation_length_ising_lattice_above_its_critical_temperature():
    """Test that 128 by 128 Ising lattices at temperatures 2.5 and 3 give the exact length within 15 percent."""
    for temperature in (2.5, 3.0):
        system = create('ising', size=128, temperature=temperature, random_state=0)
        trajectory = Simulation(system, dt=1, random_state=0).check().run(until=1500).trajectory_
        frames = np.asarray(trajectory.lattice('spins.spin'), dtype=float)[500::5]
        length = _correlation('length', frames, frames=True).length_
        assert length == pytest.approx(_exact_correlation_length(temperature), rel=0.15)


def test_correlation_length_reaction_diffusion_field():
    """Test that the stationary noisy linear reaction-diffusion field gives sqrt(D / k) = 2 within 10 percent."""
    frames = _reaction_diffusion_frames(np.random.default_rng(0), 50, 128)
    assert _correlation('length', frames, frames=True).length_ == pytest.approx(2, rel=0.1)
    assert 1 < _correlation('length', frames, frames=True, method='fit').length_ < 4


def test_correlation_length_of_a_trajectory_reads_the_last_frame():
    """Test that a field trajectory gives the length of its last frame, or of every frame."""
    trajectory = (
        Simulation(create('ising', size=32, temperature=3.0), dt=1, random_state=0)
        .check()
        .run(until=20)
        .trajectory_
    )
    frames = np.asarray(trajectory.lattice('spins.spin'), dtype=float)
    assert _correlation('length', trajectory).length_ == _correlation('length', frames[-1]).length_
    assert (
        _correlation('length', trajectory, frames=True).length_ == _correlation('length', frames, frames=True).length_
    )


def test_overlap_of_ising_replicas():
    """Test that two replicas of a 32 by 32 Ising lattice overlap at temperature 1.5 and not at temperature 5."""
    overlaps = {}
    for temperature in (1.5, 5.0):
        replicas = [
            Simulation(create('ising', size=32, temperature=temperature, initial='up'), dt=1, random_state=seed).check()
            .run(until=500)
            .trajectory_
            for seed in (0, 1)
        ]
        overlap = _correlation('overlap', replicas).overlap_
        frames = np.stack([replica.lattice('spins.spin') for replica in replicas])
        np.testing.assert_array_equal(overlap, _correlation('overlap', frames).overlap_)
        overlaps[temperature] = float(np.mean(np.abs(overlap[100:])))
    assert overlaps[1.5] > 0.9
    assert overlaps[5.0] < 0.1
    assert _correlation('overlap', [[1, -1], [1, -1]]).overlap_ == 1.0
    with pytest.raises(AnalysisError, match='2 replicas'):
        _correlation('overlap', [[1, -1], [1, -1], [1, 1]])


def test_unknown_kind_method_and_ignored_setting():
    """Test that an unknown kind or method, an ignored setting, and missing waiting times raise errors."""
    with pytest.raises(AnalysisError, match='auto, two-time, function, length'):
        CorrelationAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='moment, fit'):
        CorrelationAnalysis('length', method='guess').check()
    with pytest.raises(AnalysisError, match='does not read max_lag'):
        CorrelationAnalysis('function', max_lag=3).check()
    with pytest.raises(AnalysisError, match='waiting times and lags'):
        CorrelationAnalysis('two-time').check()


def test_cloned_analysis_has_no_results():
    """Test that a clone keeps the settings and drops the results."""
    analysis = _correlation('auto', np.arange(10.0), max_lag=2)
    copy = clone(analysis)
    assert copy.get_params() == analysis.get_params()
    assert not hasattr(copy, 'correlation_')
