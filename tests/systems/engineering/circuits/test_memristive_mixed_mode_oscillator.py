"""Test the `_memristive_mixed_mode_oscillator` module of `skcomplex.systems.engineering.circuits`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.circuits import MemristiveMixedModeOscillatorSystem


def test_memristive_mixed_mode_oscillator_settles_on_two_large_and_eleven_small_oscillations():
    """Test that the defaults give Trzaska's mixed-mode oscillation `2^11` with his period 7.816."""
    trajectory = simulate(MemristiveMixedModeOscillatorSystem())
    settled = trajectory.times > 60
    x, times = trajectory.values('circuit.x')[settled], trajectory.times[settled]
    inner = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    peaks, peak_times = x[1:-1][inner], times[1:-1][inner]
    spikes = peak_times[peaks > 30]
    period = spikes[2] - spikes[0]
    assert period == pytest.approx(7.816, rel=0.02)
    within = (peak_times >= spikes[0]) & (peak_times < spikes[2])
    assert (peaks[within] > 30).sum() == 2
    small = peaks[within][(peaks[within] > 0) & (peaks[within] < 30)]
    assert small.size == 11
