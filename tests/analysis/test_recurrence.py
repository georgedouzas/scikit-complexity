"""Test the recurrence quantification analysis of ordered, chaotic, intermittent, and random points."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from skcomplex.analysis import EmbeddingAnalysis, RecurrenceAnalysis
from skcomplex.core.errors import AnalysisError


def _recurrence(source: object, **settings: object) -> RecurrenceAnalysis:
    """Return the recurrence analysis of a source, run."""
    return RecurrenceAnalysis(**settings).check().prepare(source).run()


def _embed(values: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Return the delay vectors of a series."""
    return EmbeddingAnalysis(delay=delay, dimension=dimension).check().prepare(values).run().points_


def _lorenz_states(rho: float, step: float, duration: float) -> np.ndarray:
    """Return states of the Lorenz-63 flow every `step` after a transient of 100, one row per sample."""

    def drift(_, state):
        x, y, z = state
        return [10 * (y - x), x * (rho - z) - y, x * y - 8 / 3 * z]

    times = np.arange(100, 100 + duration, step)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 1], t_eval=times, rtol=1e-9, atol=1e-9)
    return solution.y.T


def _logistic(r: float, size: int) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = 0.3
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def test_recurrence_quantification_order_and_noise():
    """Test that a sine is deterministic, white noise is not, and the Lorenz x lies between them."""
    sine = _recurrence(_embed(np.sin(np.arange(2015) * 0.1), 2, 15))
    noise = _recurrence(np.random.default_rng(2).standard_normal(2000))
    chaos = _recurrence(_embed(_lorenz_states(28.0, 0.05, 110.0)[:2016, 0], 3, 8))
    assert sine.determinism_ > 0.95
    assert noise.determinism_ < 0.3
    assert noise.determinism_ < chaos.determinism_ < sine.determinism_
    assert sine.rate_ == pytest.approx(0.05, abs=0.01)


@pytest.mark.slow
def test_recurrence_quantification_intermittency_on_two_domains():
    """Test that intermittent logistic and Lorenz runs, sampled once per laminar orbit, are more laminar than chaos."""
    near_window = _recurrence(_logistic(3.8282, 6000)[::3][:2000])
    chaotic_map = _recurrence(_logistic(4.0, 6000)[::3][:2000])
    assert near_window.laminarity_ > chaotic_map.laminarity_
    maxima = {}
    for rho in (166.2, 28.0):
        z = _lorenz_states(rho, 0.01, 1500.0)[:, 2]
        maxima[rho] = _recurrence(z[find_peaks(z)[0]][::2][:1500]).laminarity_
    assert maxima[166.2] > maxima[28.0]


def test_recurrence_matrix_with_a_radius():
    """Test that a fixed radius gives a symmetric matrix with a true diagonal, and isolated points raise an error."""
    matrix = _recurrence([0.0, 0.1, 5.0], radius=0.5).matrix_.toarray()
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix.trace() == 3
    with pytest.raises(AnalysisError, match='No pair of points recurs'):
        _recurrence([0.0, 10.0, 20.0], radius=1.0)


def test_invalid_settings():
    """Test that a negative radius and an empty shortest line raise errors."""
    with pytest.raises(AnalysisError, match='radius=-1'):
        RecurrenceAnalysis(radius=-1.0).check()
    with pytest.raises(AnalysisError, match='min_line=0'):
        RecurrenceAnalysis(min_line=0).check()
