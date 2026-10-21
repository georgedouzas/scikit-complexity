"""Test Lyapunov spectra of flows and maps against known values."""

import time

import numpy as np
import pytest

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.models.mathematics.dynamical_systems import henon_map, logistic_map, lorenz, sprott_jerk
from skcomplex.models.physics.nonlinear_oscillators import van_der_pol


def test_lorenz_largest_exponent():
    """Test that the largest exponent of Lorenz-63 with the standard parameters is within 0.05 of 0.906."""
    exponents = lyapunov_spectrum(lorenz())
    assert exponents[0] == pytest.approx(0.906, abs=0.05)
    assert exponents[1] == pytest.approx(0.0, abs=0.02)
    assert exponents.sum() == pytest.approx(-(10 + 1 + 8 / 3), abs=0.05)


def test_logistic_map_at_four():
    """Test that the logistic map at r = 4 has exponent ln 2 within 0.01."""
    assert lyapunov_spectrum(logistic_map(r=4.0), until=100_000)[0] == pytest.approx(np.log(2), abs=0.01)


def test_henon_map_spectrum():
    """Test that the Hénon map's exponents match the published 0.419 and -1.62."""
    exponents = lyapunov_spectrum(henon_map(), until=100_000)
    assert exponents == pytest.approx([0.419, -1.623], abs=0.01)


def test_limit_cycle_is_not_chaotic():
    """Test that the Van der Pol limit cycle has a largest exponent of 0, not a positive one."""
    assert lyapunov_spectrum(van_der_pol(), 1, until=500)[0] == pytest.approx(0.0, abs=0.01)


def test_sweep_of_a_map_reuses_its_compiled_tangent_dynamics():
    """Test that ten spectra of the logistic map after a first call take under a second and match fresh values."""
    lyapunov_spectrum(logistic_map(r=3.9), until=2000)
    rates = np.linspace(3.5, 4.0, 10)
    start = time.perf_counter()
    exponents = [lyapunov_spectrum(logistic_map(r=r), until=2000)[0] for r in rates]
    assert time.perf_counter() - start < 1.0
    assert exponents[-1] == pytest.approx(lyapunov_spectrum(logistic_map(r=4.0), until=2000)[0])


def test_laws_with_absolute_values_have_a_spectrum():
    """Test that the Sprott jerk system, whose law has an absolute value, has three finite exponents of negative sum."""
    exponents = lyapunov_spectrum(sprott_jerk(), until=200)
    assert exponents.shape == (3,)
    assert np.all(np.isfinite(exponents))
    assert exponents.sum() < 0
