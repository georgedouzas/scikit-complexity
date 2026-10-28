"""Test the scaling analysis of series, cells, points, fluctuations at growing scales, and power laws."""

import math

import numpy as np
import pytest
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

from skcomplex.analysis import CoarseGrainAnalysis, ScalingAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create


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


def _density(positions: np.ndarray, side: float) -> np.ndarray:
    """Return the agents of a square of a side coarse-grained on a 64 by 64 grid."""
    return CoarseGrainAnalysis([64, 64], box=[side, side]).check().prepare(positions).run().field_


def _vicsek_densities(rng: np.random.Generator, agents: int, side: float, noise: float, steps: int) -> np.ndarray:
    """Return coarse-grained densities of Vicsek agents of speed 0.5 aligning within distance 1 in a periodic box."""
    positions = rng.random((agents, 2)) * side
    headings = rng.uniform(-np.pi, np.pi, agents)
    densities = []
    for step in range(steps):
        pairs = cKDTree(positions, boxsize=side).query_pairs(1.0, output_type='ndarray')
        sines, cosines = np.sin(headings), np.cos(headings)
        total_sines, total_cosines = sines.copy(), cosines.copy()
        for first, second in ((0, 1), (1, 0)):
            np.add.at(total_sines, pairs[:, first], sines[pairs[:, second]])
            np.add.at(total_cosines, pairs[:, first], cosines[pairs[:, second]])
        headings = np.arctan2(total_sines, total_cosines) + noise * rng.uniform(-np.pi, np.pi, agents)
        positions = np.mod(positions + 0.5 * np.column_stack([np.cos(headings), np.sin(headings)]), side)
        if step >= steps // 3 and step % 50 == 0:
            densities.append(_density(positions, side))
    return np.asarray(densities)


def test_variance_mean_of_box_counts_giant_in_ordered_flocks():
    """Test that an ideal gas has normal number fluctuations and ordered Vicsek agents giant ones."""
    rng = np.random.default_rng(2)
    gas = np.asarray([_density(rng.random((4000, 2)) * 32, 32) for _ in range(40)])
    assert _scaling('variance-mean', gas).exponent_ == pytest.approx(0.5, abs=0.05)
    assert _scaling('variance-mean', _vicsek_densities(rng, 2000, 22.6, 0.3, 2000)).exponent_ > 0.6


def test_variance_mean_against_size_of_an_ideal_lattice_gas():
    """Test that the particle number in subvolumes of a lattice gas fluctuates as the inverse root of the volume."""
    occupied = np.random.default_rng(3).random((200, 256, 256)) < 0.3
    samples = {side**2: occupied[:, :side, :side].sum(axis=(1, 2)) for side in (4, 8, 16, 32, 64)}
    relative = _scaling('variance-mean', samples, against='size')
    assert relative.exponent_ == pytest.approx(-0.5, abs=0.05)
    assert relative.scales_.tolist() == [16.0, 64.0, 256.0, 1024.0, 4096.0]
    assert _scaling('variance-mean', samples).exponent_ == pytest.approx(0.5, abs=0.05)


def test_power_exponent_of_two_mean_field_transitions():
    """Test that the mean-field Ising model and the Ott-Antonsen Kuramoto model give the exponent 1/2."""
    temperatures = np.linspace(0.95, 0.995, 10)
    magnetizations = [optimize.brentq(lambda m, t=t: m - math.tanh(m / t), 1e-6, 1.0) for t in temperatures]
    power = _scaling('power', magnetizations, parameters=temperatures, critical=1.0)
    assert power.exponent_[0] == pytest.approx(0.5, abs=0.05)
    couplings = np.linspace(1.02, 1.2, 10)
    orders = [
        Simulation(create('ott_antonsen', r=0.5, K=coupling, Delta=0.5), dt=1.0).check().run(until=400.0).trajectory_
        for coupling in couplings
    ]
    names = orders[0].problem.states.names
    final = [float(np.abs(trajectory.values(names[0])[-1])) for trajectory in orders]
    power = _scaling('power', final, parameters=couplings, critical=1.0, above=True)
    assert power.exponent_[0] == pytest.approx(0.5, abs=0.05)
    with pytest.raises(AnalysisError, match='chosen side'):
        _scaling('power', [0.1, 0.2], parameters=[1.5, 2.0], critical=1.0)


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='detrended-fluctuation, rescaled-range'):
        ScalingAnalysis('wavelet').check()
    with pytest.raises(AnalysisError, match='does not read order'):
        ScalingAnalysis('box-counting', order=2).check()
    with pytest.raises(AnalysisError, match="against='variance'"):
        ScalingAnalysis('variance-mean', against='variance').check()
    with pytest.raises(AnalysisError, match='critical value'):
        ScalingAnalysis('power', parameters=[1.0, 2.0]).check()
