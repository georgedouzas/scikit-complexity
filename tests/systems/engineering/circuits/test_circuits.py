"""Test the `_circuits` module of `skcomplex.systems.engineering.circuits`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.engineering.circuits import (
    chua,
    impulsive_circuits,
    memristor_oscillator,
    mixed_mode_chua,
    tunnel_diode_oscillator,
)


def _peaks(series):
    """Return the values of the local maxima of a series."""
    inner = series[1:-1]
    return inner[(inner > series[:-2]) & (inner > series[2:])]


def test_chua_double_scroll_visits_both_scrolls():
    """Test that the double-scroll attractor stays bounded and visits both sides of the origin."""
    trajectory = Simulation(chua(), dt=0.05).prepare().run(until=100).trajectory_
    x = trajectory.values('circuit.x')[400:]
    assert np.abs(x).max() < 3.0
    assert 0.2 < (x > 0).mean() < 0.8


def test_chua_sine_diode_has_several_scrolls():
    """Test that the sine diode gives a bounded attractor spanning more than two scrolls."""
    trajectory = Simulation(chua(nonlinearity='sine'), dt=0.5).prepare().run(until=600).trajectory_
    x = trajectory.values('circuit.x')[100:]
    assert np.abs(x).max() < 15.0
    assert x.min() < -4 * 1.3
    assert x.max() > 4 * 1.3


def test_chua_rejects_unknown_nonlinearity():
    """Test that an unknown diode characteristic raises a parameter error."""
    with pytest.raises(ParameterError, match='Unknown nonlinearity'):
        chua(nonlinearity='cubic')


@pytest.mark.parametrize(('c', 'mixed'), [(0.00053, True), (0.00105, False)])
def test_mixed_mode_chua_oscillations(c, mixed):
    """Test that small oscillations come between the spikes for `c = 0.00053` and not for `c = 0.00105`."""
    solver = ScipyODESolver(method='Radau')
    trajectory = Simulation(mixed_mode_chua(c=c), dt=0.1, solver=solver).prepare().run(until=50).trajectory_
    peaks = _peaks(trajectory.values('circuit.x')[100:])
    assert (peaks > 1.0).sum() >= 4
    assert ((peaks < 0.3).sum() >= 4) == mixed


def test_memristor_oscillator_keeps_oscillating():
    """Test that the memristor circuit neither decays nor diverges."""
    trajectory = Simulation(memristor_oscillator(), dt=0.1).prepare().run(until=200).trajectory_
    voltage = trajectory.values('circuit.v')
    assert np.abs(voltage).max() < 10.0
    assert np.ptp(voltage[-500:]) > 0.5


def test_tunnel_diode_limit_cycle_amplitude():
    """Test that the tunnel-diode oscillator settles on the van der Pol limit cycle of amplitude `2 sqrt(alpha / 3)`."""
    trajectory = Simulation(tunnel_diode_oscillator(alpha=1.0, gamma=1.0), dt=0.05).prepare().run(until=50).trajectory_
    assert np.abs(trajectory.values('circuit.V')[-300:]).max() == pytest.approx(2 * math.sqrt(1 / 3), rel=0.02)


def test_impulsive_circuit_kicks_the_rate():
    """Test that every impulse fires once per period and raises the rate by `A / (L C)`."""
    system = impulsive_circuits(A=0.5, L=2.0, C=0.5, T=2.0)
    rate = PassCompiler().prepare(system).run().problem_.states.names.index('circuit.x_d1')
    trajectory = Simulation(system, dt=0.05).prepare().run(until=10).trajectory_
    events = [event for event in trajectory.events if event.label == 'impulse']
    assert [round(event.time, 6) for event in events] == [2.0, 4.0, 6.0, 8.0, 10.0][: len(events)]
    assert len(events) >= 4
    assert all(event.after[rate] - event.before[rate] == pytest.approx(0.5) for event in events)


def test_circuit_and_oscillator_agree():
    """Test that the RLC circuit and the damped oscillator agree to the solver tolerance for equal coefficients."""
    charge = (
        create_simulation('rlc_circuit', L=2.0, R=0.4, C=1 / 8, V=0.5, omega=1.3)
        .prepare()
        .run()
        .trajectory_.values('circuit.q')
    )
    displacement = (
        create_simulation('driven_damped_oscillator', m=2.0, c=0.4, k=8.0, F=0.5, omega=1.3)
        .prepare()
        .run()
        .trajectory_.values('mass.x')
        .ravel()
    )
    assert np.allclose(charge, displacement, rtol=1e-5, atol=1e-5)


def _phase(y, z):
    """Return the unwrapped angle of an orbit in the plane of two variables, around their means."""
    return np.unwrap(np.arctan2(z - z.mean(), y - y.mean()))


def test_memristive_mixed_mode_oscillator_settles_on_two_large_and_eleven_small_oscillations():
    """Test that the defaults give Trzaska's mixed-mode oscillation `2^11` with a period near 7.8."""
    trajectory = create_simulation('memristive_mixed_mode_oscillator').prepare().run().trajectory_
    settled = trajectory.times > 60
    x, times = trajectory.values('circuit.x')[settled], trajectory.times[settled]
    inner = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    peaks, peak_times = x[1:-1][inner], times[1:-1][inner]
    spikes = peak_times[peaks > 30]
    period = spikes[2] - spikes[0]
    assert period == pytest.approx(7.8, rel=0.03)
    within = (peak_times >= spikes[0]) & (peak_times < spikes[2])
    assert (peaks[within] > 30).sum() == 2
    small = peaks[within][(peaks[within] > 0) & (peaks[within] < 30)]
    assert small.size == 11


@pytest.mark.parametrize(('k', 'locked'), [(0.0, False), (0.4, True)])
def test_coupled_colpitts_lock_their_phases_for_moderate_coupling(k, locked):
    """Test that the phases of the detuned oscillators drift apart without coupling and stay together for `k = 0.4`."""
    trajectory = create_simulation('coupled_colpitts', k=k).prepare().run(until=400).trajectory_
    settled = trajectory.times > 100
    phases = [
        _phase(trajectory.values(f'oscillator_{index}.y')[settled], trajectory.values(f'oscillator_{index}.z')[settled])
        for index in range(2)
    ]
    assert (np.ptp(phases[0] - phases[1]) < 2 * math.pi) == locked
