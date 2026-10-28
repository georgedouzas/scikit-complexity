"""Test delay embeddings, fractal dimensions, and recurrences."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from skcomplex.analysis import (
    correlation_dimension,
    delay_embedding,
    embed,
    recurrence_matrix,
    recurrence_quantification,
)
from skcomplex.core.errors import AnalysisError


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


def _logistic(r: float, size: int) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = 0.3
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def test_embed_lorenz():
    """Test that 20,000 samples of the Lorenz x every 0.01 give a delay between 0.10 and 0.20 and dimension 3."""
    embedding = embed(_lorenz_states(28.0, 0.01, 200.0)[:20_000, 0], max_delay=60)
    assert 0.10 <= embedding.delay * 0.01 <= 0.20
    assert embedding.dimension == 3
    assert embedding.points.shape[1] == 3


def test_embed_map():
    """Test that 10,000 iterates of the Hénon x give dimension 2."""
    assert embed(_henon(10_000)[:, 0], delay=1).dimension == 2


def test_delay_embedding_rejects_short_series():
    """Test that a series too short for one vector raises an error, as does a dimension never reached."""
    with pytest.raises(AnalysisError, match='no delay vector'):
        delay_embedding([1.0, 2.0], 3, 2)
    with pytest.raises(AnalysisError, match='No dimension up to 1'):
        embed(np.random.default_rng(0).standard_normal(500), delay=1, max_dimension=1)


def test_correlation_dimension_of_attractors_and_agents():
    """Test the correlation dimensions of the Lorenz and Hénon attractors and of uniform agents in a square."""
    assert correlation_dimension(_lorenz_states(28.0, 0.05, 1000.0)[:20_000]).dimension == pytest.approx(2.05, abs=0.1)
    assert correlation_dimension(_henon(20_000)).dimension == pytest.approx(1.21, abs=0.05)
    agents = np.random.default_rng(1).random((5000, 2))
    assert correlation_dimension(agents).dimension == pytest.approx(2, abs=0.05)
    with pytest.raises(AnalysisError, match='Fewer than 2 radii'):
        correlation_dimension(agents, [10.0, 20.0])




def test_recurrence_quantification_order_and_noise():
    """Test that a sine is deterministic, white noise is not, and the Lorenz x lies between them."""
    sine = recurrence_quantification(delay_embedding(np.sin(np.arange(2015) * 0.1), 2, 15))
    noise = recurrence_quantification(np.random.default_rng(2).standard_normal(2000))
    chaos = recurrence_quantification(delay_embedding(_lorenz_states(28.0, 0.05, 110.0)[:2016, 0], 3, 8))
    assert sine.determinism > 0.95
    assert noise.determinism < 0.3
    assert noise.determinism < chaos.determinism < sine.determinism
    assert sine.rate == pytest.approx(0.05, abs=0.01)


@pytest.mark.slow
def test_recurrence_quantification_intermittency_on_two_domains():
    """Test that intermittent logistic and Lorenz runs, sampled once per laminar orbit, are more laminar than chaos."""
    near_window = recurrence_quantification(_logistic(3.8282, 6000)[::3][:2000])
    chaotic_map = recurrence_quantification(_logistic(4.0, 6000)[::3][:2000])
    assert near_window.laminarity > chaotic_map.laminarity
    maxima = {}
    for rho in (166.2, 28.0):
        z = _lorenz_states(rho, 0.01, 1500.0)[:, 2]
        maxima[rho] = recurrence_quantification(z[find_peaks(z)[0]][::2][:1500]).laminarity
    assert maxima[166.2] > maxima[28.0]


def test_recurrence_matrix_with_a_radius():
    """Test that a fixed radius gives a symmetric matrix with a true diagonal, and isolated points raise an error."""
    matrix = recurrence_matrix([0.0, 0.1, 5.0], 0.5).toarray()
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix.trace() == 3
    with pytest.raises(AnalysisError, match='No pair of points recurs'):
        recurrence_quantification([0.0, 10.0, 20.0], 1.0)
