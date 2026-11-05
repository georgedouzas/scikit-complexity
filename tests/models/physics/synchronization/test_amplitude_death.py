"""Test the `_amplitude_death` module of `skcomplex.models.physics.synchronization`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models import describe
from skcomplex.models.physics.synchronization import heterogeneous_global_coupling, oscillator_death


def _decay_rate(times, values, start):
    """Return the slope of the logarithm of positive values after a start time."""
    selected = times >= start
    return np.polyfit(times[selected], np.log(values[selected]), 1)[0]


def test_oscillator_death_follows_lorentzian_threshold():
    """Test that oscillations die at the rate `ω0 - 1` above the critical width 1 and persist below it."""
    dying = (
        Simulation(
            built := oscillator_death(100, K=3.0, width=1.5),
            dt=0.5,
            observables=describe('oscillator_death').observables_of(built),
        )
        .prepare()
        .run(until=20)
        .trajectory_
    )
    amplitude = np.asarray(dying.values('amplitude'))
    rate = _decay_rate(np.asarray(dying.times), amplitude, start=8.0)
    assert amplitude[-1] < 1e-3
    assert rate == pytest.approx(1 - 1.5, rel=0.1)
    living = (
        Simulation(
            built := oscillator_death(100, K=3.0, width=0.5),
            dt=0.5,
            observables=describe('oscillator_death').observables_of(built),
        )
        .prepare()
        .run(until=20)
        .trajectory_
    )
    assert living.values('amplitude')[-1] > 0.3


def test_oscillator_death_follows_quartic_threshold():
    """Test that the quartic density kills the oscillations above `(1 + √(2K - 1)) / √2` and not below it."""
    critical = (1 + math.sqrt(2 * 2.0 - 1)) / math.sqrt(2)
    above = oscillator_death(100, K=2.0, width=1.2 * critical, distribution='quartic')
    below = oscillator_death(100, K=2.0, width=0.8 * critical, distribution='quartic')
    assert (
        Simulation(above, dt=0.5, observables=describe('oscillator_death').observables_of(above))
        .prepare()
        .run(until=40)
        .trajectory_.values('amplitude')[-1]
        < 1e-3
    )
    assert (
        Simulation(below, dt=0.5, observables=describe('oscillator_death').observables_of(below))
        .prepare()
        .run(until=40)
        .trajectory_.values('amplitude')[-1]
        > 0.3
    )


def test_oscillator_death_rejects_unknown_distribution():
    """Test that an unknown density of frequencies raises a validation error that names the choices."""
    with pytest.raises(ValidationError, match='lorentzian'):
        oscillator_death(10, distribution='gaussian')


@pytest.mark.slow
def test_heterogeneous_global_coupling_restabilizes_fixed_point():
    """Test that intermediate coupling kills the oscillations, while weak and very strong coupling do not."""
    amplitudes = {
        k: Simulation(
            built := heterogeneous_global_coupling(100, k=k),
            dt=0.5,
            observables=describe('heterogeneous_global_coupling').observables_of(built),
        )
        .prepare()
        .run(until=60)
        .trajectory_.values('amplitude')[-1]
        for k in (0.0, 2.0, 15.0)
    }
    assert amplitudes[0.0] > 0.1
    assert amplitudes[2.0] < 1e-3
    assert amplitudes[15.0] > 0.01


def test_heterogeneous_global_coupling_decays_as_order_parameters_predict():
    """Test that the decay rate at k = 4 matches the leading eigenvalue of the closed order-parameter equations."""
    system = heterogeneous_global_coupling(100, k=4.0)
    taus = system.elements[0].values['tau']
    nu, variance = complex(0.1, 2.0), float(np.var(taus))
    roots = [nu - 2.0 + sign * np.sqrt(4.0 + variance * nu**2) for sign in (1, -1)]
    predicted = max(root.real for root in roots)
    trajectory = (
        Simulation(system, dt=0.5, observables=describe('heterogeneous_global_coupling').observables_of(system))
        .prepare()
        .run(until=60)
        .trajectory_
    )
    rate = _decay_rate(np.asarray(trajectory.times), np.asarray(trajectory.values('amplitude')), start=20.0)
    assert predicted < 0
    assert rate == pytest.approx(predicted, rel=0.15)
