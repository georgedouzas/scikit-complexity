"""Test the `_diluted` module of `skcomplex.systems.computer_science.neural_networks`."""

import numpy as np
import pytest
from scipy import integrate

from ...conftest import _series


def _hopfield_map(m, connectivity, n_patterns, beta):
    """Return the overlap after one step of the annealed approximation, equation (6.55) of Bertin."""
    spread = np.sqrt(connectivity * (n_patterns - m**2))

    def integrand(z):
        return np.exp(-(z**2) / 2) * np.tanh(beta * (connectivity * m + spread * z)) / np.sqrt(2 * np.pi)

    return integrate.quad(integrand, -10, 10)[0]


def test_diluted_hopfield_reaches_the_annealed_fixed_point():
    """Test that the overlap settles at the fixed point of the annealed map of Bertin within 0.05."""
    overlap = _series('diluted_hopfield', 20).values('overlap')
    m = 0.3
    for _ in range(50):
        m = _hopfield_map(m, 50.0, 10, 0.06)
    assert np.mean(overlap[-5:]) == pytest.approx(m, abs=0.05)
