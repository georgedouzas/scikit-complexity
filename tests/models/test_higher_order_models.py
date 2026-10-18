"""Test the library models written in their natural order across fields."""

import numpy as np
import pytest

from skcomplex.analysis import equations_of
from skcomplex.models import create, run


def test_circuit_and_oscillator_agree():
    """Test that the RLC circuit and the damped oscillator give the same trajectory for matching coefficients."""
    charge = run('rlc_circuit', L=2.0, R=0.4, C=1 / 8, V=0.5, omega=1.3).trajectory_['circuit.q']
    displacement = run('driven_damped_oscillator', m=2.0, c=0.4, k=8.0, F=0.5, omega=1.3).trajectory_['mass.x']
    assert np.allclose(charge, displacement, rtol=1e-12, atol=1e-12)


def test_pendulum_equation_is_newtons_law():
    """Test that the pendulum is compiled from Newton's law into one second-order equation."""
    equations = equations_of(create('pendulum'))
    assert list(equations) == ['bob.theta']
    assert 'sin(theta_bob(t))' in str(equations['bob.theta'].rhs())


def test_fibonacci_rabbits():
    """Test that Fibonacci's rabbits double their growth like the Fibonacci numbers."""
    assert run('fibonacci_rabbits', settings={'until': 9}).trajectory_['rabbits.pairs'].tolist() == [
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
    ]


def test_jerk_is_bounded():
    """Test that Sprott's jerk equation stays on its bounded attractor."""
    x = run('sprott_jerk').trajectory_['jerk.x']
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10


@pytest.mark.parametrize('cv', [0.6, 1.2])
def test_multiplier_accelerator_stability(cv):
    """Test that the cycle damps when `c v < 1` and grows when `c v > 1`."""
    income = run('multiplier_accelerator', v=cv / 0.8, settings={'until': 60}).trajectory_['economy.Y']
    deviation = np.abs(income - 100.0)
    assert (deviation[-10:].max() > deviation[:10].max()) == (cv > 1)
