"""Test Lyapunov spectra of flows and maps against known values."""

import numpy as np
import pytest

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.models.nonlinear import henon_map, logistic_map, lorenz, van_der_pol


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
