"""Test the scaling analysis of series, cells, and points, and Binder cumulants of the Ising transition."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.analysis import ScalingAnalysis, binder_crossings, binder_cumulant
from skcomplex.core.errors import AnalysisError

from .conftest import ISING_SIZES, ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))


def _scaling(kind: str, source: object, **settings: object) -> ScalingAnalysis:
    """Return the scaling analysis of a source for one kind, run."""
    return ScalingAnalysis(kind, **settings).check().prepare(source).run()


def _fractional_gaussian_noise(size: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Return fractional Gaussian noise by the circulant embedding of Davies and Harte."""
    lags = np.arange(size + 1, dtype=float)
    covariance = 0.5 * (np.abs(lags + 1) ** (2 * hurst) - 2 * lags ** (2 * hurst) + np.abs(lags - 1) ** (2 * hurst))
    circulant = np.concatenate([covariance, covariance[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(circulant).real, 0, None)
    weights = np.sqrt(eigenvalues / circulant.size) * (
        rng.standard_normal(circulant.size) + 1j * rng.standard_normal(circulant.size)
    )
    return np.fft.fft(weights).real[:size]


def _sierpinski(size: int) -> np.ndarray:
    """Return a Sierpinski triangle as Pascal's triangle modulo 2."""
    rows, columns = np.indices((size, size))
    return (rows & columns) == columns


def _lorenz_states(rho: float, step: float, duration: float) -> np.ndarray:
    """Return states of the Lorenz-63 flow every `step` after a transient of 100, one row per sample."""

    def drift(_, state):
        x, y, z = state
        return [10 * (y - x), x * (rho - z) - y, x * y - 8 / 3 * z]

    times = np.arange(100, 100 + duration, step)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 1], t_eval=times, rtol=1e-9, atol=1e-9)
    return solution.y.T


def _henon(size: int) -> np.ndarray:
    """Return iterates of the Hénon map after a transient of 100, one row per iterate."""
    points = np.empty((size + 100, 2))
    x, y = 0.1, 0.0
    for index in range(points.shape[0]):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        points[index] = x, y
    return points[100:]


def test_hurst_exponent_white_noise_memory():
    """Test that white noise has a Hurst exponent within 0.05 of 0.5 by both estimators."""
    noise = np.random.default_rng(4).standard_normal(20_000)
    assert _scaling('detrended-fluctuation', noise).exponent_[0] == pytest.approx(0.5, abs=0.05)
    assert _scaling('rescaled-range', noise).exponent_[0] == pytest.approx(0.5, abs=0.05)


@pytest.mark.slow
def test_white_noise_has_no_memory():
    """Test that white noise has a Hurst exponent within 0.05 of 0.5 and integrated noise near 1.5."""
    noise = np.random.default_rng(6).standard_normal(50_000)
    assert _scaling('detrended-fluctuation', noise).exponent_[0] == pytest.approx(0.5, abs=0.05)
    assert _scaling('detrended-fluctuation', np.cumsum(noise)).exponent_[0] == pytest.approx(1.5, abs=0.1)


def test_detrended_fluctuation_persistent_and_integrated_noise():
    """Test that fractional Gaussian noise with H = 0.7 gives 0.7 and a random walk 1.5, within 0.05."""
    persistent = _fractional_gaussian_noise(20_000, 0.7, np.random.default_rng(11))
    walk = np.cumsum(np.random.default_rng(5).standard_normal(20_000))
    exponents = _scaling('detrended-fluctuation', np.column_stack([persistent, walk])).exponent_
    assert exponents == pytest.approx([0.7, 1.5], abs=0.05)


def test_detrended_fluctuation_rejects_short_series():
    """Test that short series raise an error."""
    with pytest.raises(AnalysisError, match='at least 100'):
        _scaling('detrended-fluctuation', np.zeros(10))


def test_sierpinski_dimension():
    """Test that the box-counting dimension of a Sierpinski triangle is within 0.05 of log 3 / log 2."""
    assert _scaling('box-counting', _sierpinski(1024)).exponent_ == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_correlation_dimension_of_attractors_and_agents():
    """Test the correlation dimensions of the Lorenz and Hénon attractors and of uniform agents in a square."""
    lorenz = _lorenz_states(28.0, 0.05, 1000.0)[:20_000]
    assert _scaling('correlation-dimension', lorenz).exponent_ == pytest.approx(2.05, abs=0.1)
    assert _scaling('correlation-dimension', _henon(20_000)).exponent_ == pytest.approx(1.21, abs=0.05)
    agents = np.random.default_rng(1).random((5000, 2))
    assert _scaling('correlation-dimension', agents).exponent_ == pytest.approx(2, abs=0.05)
    with pytest.raises(AnalysisError, match='Fewer than 2 radii'):
        _scaling('correlation-dimension', agents, scales=[10.0, 20.0])


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='detrended-fluctuation, rescaled-range'):
        ScalingAnalysis('wavelet').check()
    with pytest.raises(AnalysisError, match='does not read order'):
        ScalingAnalysis('box-counting', order=2).check()


def test_binder_crossings_ising_critical_point(ising_magnetizations):
    """Test that the Binder cumulants of Ising lattices of sides 16, 32, and 64 cross within 1 percent of T_c."""
    cumulants = {side: [binder_cumulant(samples) for samples in ising_magnetizations[side]] for side in ISING_SIZES}
    crossings = binder_crossings(ISING_TEMPERATURES, cumulants)
    np.testing.assert_allclose(crossings, CRITICAL, rtol=0.01)
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        binder_crossings(ISING_TEMPERATURES, {16: cumulants[16]})
