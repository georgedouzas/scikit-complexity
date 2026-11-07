"""Test the `_neural_masses` module of `skcomplex.systems.biology.neuroscience`."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.neuroscience import next_generation_neural_mass

from ..conftest import _run_until


def _gamma_peaks(rates: np.ndarray) -> np.ndarray:
    """Return the peaks of the firing rate over the last half of a run sampled every 0.1 ms."""
    peaks, _ = find_peaks(rates[rates.size // 2 :], prominence=0.01)
    return peaks


def test_ping_rests_below_and_oscillates_in_gamma_above_the_hopf_bifurcation():
    """Test Segneri et al.: the unforced PING mass rests for `H_e = -5` and oscillates near 45 Hz for `H_e = 10`."""
    rest = _run_until(next_generation_neural_mass(H_e=-5.0, I0=0.0), 1000, dt=0.1).values('excitatory.r')
    assert np.ptp(rest[5000:]) < 1e-4
    gamma = _run_until(next_generation_neural_mass(H_e=10.0, I0=0.0), 1000, dt=0.1).values('excitatory.r')
    assert 40 <= _gamma_peaks(gamma).size / 0.5 <= 50


def test_theta_forcing_nests_gamma_bursts_in_the_theta_cycle():
    """Test that the theta current drives several gamma cycles per theta cycle in the PING mass."""
    forced = _run_until(next_generation_neural_mass(), 1000, dt=0.1).values('excitatory.r')
    assert 3 <= _gamma_peaks(forced).size / 2.5 <= 10


def test_ing_oscillates_through_its_exponential_synapses():
    """Test that the inhibitory mass rests at `H = 2` without forcing and oscillates at `H = 10`."""
    rest = _run_until(next_generation_neural_mass('ing', I0=0.0), 600, dt=0.1).values('inhibitory.r')
    assert np.ptp(rest[3000:]) < 1e-4
    rhythm = _run_until(next_generation_neural_mass('ing', I0=0.0, H_i=10.0), 600, dt=0.1).values('inhibitory.r')
    assert _gamma_peaks(rhythm).size > 10


def test_unknown_setup_raises():
    """Test that an unknown setup raises an error."""
    with pytest.raises(ValidationError):
        next_generation_neural_mass('pong')
