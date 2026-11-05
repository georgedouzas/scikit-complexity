"""Test the `_expansion` module of `skcomplex.systems.physics.cosmology`."""

import numpy as np
import pytest

from skcomplex.systems.physics.cosmology import friedmann_expansion

from ..conftest import _run, _series


def test_einstein_de_sitter_universe_grows_as_two_thirds_power():
    """Test that a flat universe of matter expands as `(1 + 3 H0 t / 2)^(2/3)`."""
    trajectory = _run(friedmann_expansion(omega_matter=1.0, omega_lambda=0.0), 3.0, 0.1)
    expected = (1 + 1.5 * trajectory.times) ** (2 / 3)
    assert _series(trajectory, 'universe.a')[0] == pytest.approx(expected, rel=1e-8)


def test_dark_energy_universe_keeps_the_friedmann_equation_and_accelerates():
    """Test that the universe with dark energy keeps the Friedmann equation as a first integral and accelerates."""
    trajectory = _run(friedmann_expansion(), 2.0, 0.05)
    a, rate = _series(trajectory, 'universe.a', 'universe.a_d1')
    friedmann = rate**2 - (0.3 / a + 0.7 * a**2)
    assert np.abs(friedmann).max() < 1e-8
    assert np.all(np.diff(rate) > 0)
