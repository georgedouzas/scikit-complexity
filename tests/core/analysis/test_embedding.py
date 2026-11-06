"""Test the delay embedding analysis and its choice of delay and dimension."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.analysis import EmbeddingAnalysis
from skcomplex.core.errors import AnalysisError


def _embedding(kind: str, source: object, **settings: object) -> EmbeddingAnalysis:
    """Return the embedding analysis of a source for one kind, run."""
    return EmbeddingAnalysis(kind, **settings).prepare(source).run()


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


def test_embed_lorenz():
    """Test that 20,000 samples of the Lorenz x every 0.01 give a delay between 0.10 and 0.25 and dimension 3."""
    embedding = _embedding('delay', _lorenz_states(28.0, 0.01, 200.0)[:20_000, 0], max_delay=60)
    assert 0.10 <= embedding.delay_ * 0.01 <= 0.25
    assert embedding.dimension_ == 3
    assert embedding.points_.shape[1] == 3
    assert embedding.mutual_information_.size == 61


def test_embed_map():
    """Test that 10,000 iterates of the Hénon x give dimension 2."""
    assert _embedding('delay', _henon(10_000)[:, 0], delay=1).dimension_ == 2


def test_false_nearest_neighbors_of_a_sine():
    """Test that a sine has no false neighbours from dimension 2."""
    fractions = _embedding('false-nearest-neighbors', np.sin(np.arange(3000) * 0.05), delay=30, max_dimension=3)
    assert fractions.false_neighbors_.round(2).tolist()[1:] == [0.0, 0.0]


def test_delay_embedding_rejects_short_series_and_several_series():
    """Test that a series too short for one vector, a dimension never reached, and several series raise errors."""
    with pytest.raises(AnalysisError, match='no delay vector'):
        _embedding('delay', [1.0, 2.0], dimension=3, delay=2)
    with pytest.raises(AnalysisError, match='No dimension up to 1'):
        _embedding('delay', np.random.default_rng(0).standard_normal(500), delay=1, max_dimension=1)
    with pytest.raises(AnalysisError, match='Pick one'):
        _embedding('delay', np.zeros((10, 2)), delay=1, dimension=1)


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='delay, false-nearest-neighbors'):
        EmbeddingAnalysis('unknown').prepare()
    with pytest.raises(AnalysisError, match='does not read ratio'):
        EmbeddingAnalysis('delay', ratio=5.0).prepare()
