"""Test the `_neural_masses` module of `skcomplex.systems.biology.neuroscience`."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.neuroscience import (
    NeuralMassElement,
    NeuralMassInteraction,
    NextGenerationNeuralMassSystem,
)

from ..conftest import _run_until


def _gamma_peaks(rates: np.ndarray) -> np.ndarray:
    """Return the peaks of the firing rate over the last half of a run sampled every 0.1 ms."""
    peaks, _ = find_peaks(rates[rates.size // 2 :], prominence=0.01)
    return peaks


def test_ping_rests_below_and_oscillates_in_gamma_above_the_hopf_bifurcation():
    """Test Segneri et al.: the unforced PING mass rests for `H_e = -5` and oscillates near 45 Hz for `H_e = 10`."""
    rest = _run_until(NextGenerationNeuralMassSystem(H_e=-5.0, I0=0.0), 1000, dt=0.1).get_values('excitatory.r')
    assert np.ptp(rest[5000:]) < 1e-4
    gamma = _run_until(NextGenerationNeuralMassSystem(H_e=10.0, I0=0.0), 1000, dt=0.1).get_values('excitatory.r')
    assert 40 <= _gamma_peaks(gamma).size / 0.5 <= 50


def test_theta_forcing_nests_gamma_bursts_in_the_theta_cycle():
    """Test that the theta current drives several gamma cycles per theta cycle in the PING mass."""
    forced = _run_until(NextGenerationNeuralMassSystem(), 1000, dt=0.1).get_values('excitatory.r')
    assert 3 <= _gamma_peaks(forced).size / 2.5 <= 10


def test_ing_oscillates_through_its_exponential_synapses():
    """Test that the inhibitory mass rests at `H = 2` without forcing and oscillates at `H = 10`."""
    rest = _run_until(NextGenerationNeuralMassSystem('ing', I0=0.0), 600, dt=0.1).get_values('inhibitory.r')
    assert np.ptp(rest[3000:]) < 1e-4
    rhythm = _run_until(NextGenerationNeuralMassSystem('ing', I0=0.0, H_i=10.0), 600, dt=0.1).get_values('inhibitory.r')
    assert _gamma_peaks(rhythm).size > 10


def test_unknown_setup_raises():
    """Test that an unknown setup raises an error."""
    with pytest.raises(ValidationError):
        NextGenerationNeuralMassSystem('pong').prepare()


def test_mean_field_interaction_gives_the_montbrio_rates():
    """Test that the interaction gives `r`, `v`, and `s` the exact mean-field rates of one population."""
    population = NeuralMassElement('e', r=0.02, v=-1.5, s=0.01, tau=20.0, delta=1.0, H=-5.0, tau_d=10.0)
    system = System(elements=[population], interactions=[NeuralMassInteraction('mean_field', target='e')])
    problem = system.prepare().simulate(until=0).problem
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = [
        1.0 / (20.0**2 * np.pi) + 2 * 0.02 * -1.5 / 20.0,
        ((-1.5) ** 2 - 5.0) / 20.0 - 20.0 * (np.pi * 0.02) ** 2,
        (0.02 - 0.01) / 10.0,
    ]
    assert np.asarray(rates)[:3] == pytest.approx(expected)
