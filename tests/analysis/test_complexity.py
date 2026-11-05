"""Test the complexity-entropy plane."""

import numpy as np
import pytest

from skcomplex.analysis import ComplexityAnalysis
from skcomplex.errors import AnalysisError


def _logistic(r: float, size: int, start: float = 0.3) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = start
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def test_complexity_entropy_plane():
    """Test that noise, the logistic map at r = 4, and a sine take their places between the bounds of the plane."""
    series = np.column_stack(
        [np.random.default_rng(5).standard_normal(200_000), _logistic(4.0, 200_000), np.sin(np.arange(200_000) * 0.05)],
    )
    plane = ComplexityAnalysis(dimension=6).prepare(series).run()
    complexity = plane.complexity_
    assert plane.entropy_[0] > 0.99
    assert complexity[0] < 0.02
    assert complexity[1] > 0.3
    for entropy, value in zip(plane.entropy_, complexity, strict=True):
        low = np.interp(entropy, *plane.minimum_[np.argsort(plane.minimum_[:, 0])].T)
        high = np.interp(entropy, *plane.maximum_[np.argsort(plane.maximum_[:, 0])].T)
        assert low - 1e-6 <= value <= high + 1e-6


def test_complexity_entropy_plane_of_a_field_and_its_checks():
    """Test that a window reads the frames of a field, and that a pattern length below 1 is rejected."""
    field = np.random.default_rng(0).standard_normal((2, 64, 64))
    assert ComplexityAnalysis(dimension=(2, 2)).prepare(field).run().entropy_.shape == (2,)
    with pytest.raises(AnalysisError, match='Use 1 or more'):
        ComplexityAnalysis(dimension=0).prepare()
