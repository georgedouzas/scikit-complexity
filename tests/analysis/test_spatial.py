"""Test the `_spatial` module of `skcomplex.analysis`."""

import numpy as np
import pytest

from skcomplex.analysis import (
    box_counting_dimension,
    correlation_function,
)
from skcomplex.core.errors import AnalysisError


def _sierpinski(size: int) -> np.ndarray:
    """Return a Sierpinski triangle as Pascal's triangle modulo 2."""
    rows, columns = np.indices((size, size))
    return (rows & columns) == columns


def test_sierpinski_dimension():
    """Test that the box-counting dimension of a Sierpinski triangle is within 0.05 of log 3 / log 2."""
    assert box_counting_dimension(_sierpinski(1024)) == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_correlation_length_of_smoothed_noise():
    """Test that smoothing noise over neighbours makes nearby cells correlated and distant ones not."""
    noise = np.random.default_rng(1).normal(size=(128, 128))
    smooth = sum(np.roll(np.roll(noise, dx, 0), dy, 1) for dx in range(-2, 3) for dy in range(-2, 3))
    _, values = correlation_function(smooth, 10)
    assert values[1] > 0.6
    assert abs(values[10]) < 0.1
    with pytest.raises(AnalysisError, match='constant'):
        correlation_function(np.ones((4, 4)))
