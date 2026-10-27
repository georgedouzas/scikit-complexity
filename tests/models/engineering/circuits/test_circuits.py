"""Test the `_circuits` module of `skcomplex.models.engineering.circuits`."""

import math

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models import run
from skcomplex.models.engineering.circuits import (
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
    trajectory = Simulation(chua(), dt=0.05).run(until=100).trajectory_
    x = trajectory.values('circuit.x')[400:]
    assert np.abs(x).max() < 3.0
    assert 0.2 < (x > 0).mean() < 0.8


def test_chua_sine_diode_has_several_scrolls():
    """Test that the sine diode gives a bounded attractor spanning more than two scrolls."""
    trajectory = Simulation(chua(nonlinearity='sine'), dt=0.5).run(until=600).trajectory_
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
    trajectory = Simulation(mixed_mode_chua(c=c), dt=0.1, solver=solver).run(until=50).trajectory_
    peaks = _peaks(trajectory.values('circuit.x')[100:])
    assert (peaks > 1.0).sum() >= 4
    assert ((peaks < 0.3).sum() >= 4) == mixed


def test_memristor_oscillator_keeps_oscillating():
    """Test that the memristor circuit neither decays nor diverges."""
    trajectory = Simulation(memristor_oscillator(), dt=0.1).run(until=200).trajectory_
    voltage = trajectory.values('circuit.v')
    assert np.abs(voltage).max() < 10.0
    assert np.ptp(voltage[-500:]) > 0.5


def test_tunnel_diode_limit_cycle_amplitude():
    """Test that the tunnel-diode oscillator settles on the van der Pol limit cycle of amplitude `2 sqrt(alpha / 3)`."""
    trajectory = Simulation(tunnel_diode_oscillator(alpha=1.0, gamma=1.0), dt=0.05).run(until=50).trajectory_
    assert np.abs(trajectory.values('circuit.V')[-300:]).max() == pytest.approx(2 * math.sqrt(1 / 3), rel=0.02)


def test_impulsive_circuit_kicks_the_rate():
    """Test that every impulse fires once per period and raises the rate by `A / (L C)`."""
    system = impulsive_circuits(A=0.5, L=2.0, C=0.5, T=2.0)
    rate = compile_system(system).states.names.index('circuit.x_d1')
    trajectory = Simulation(system, dt=0.05).run(until=10).trajectory_
    events = [event for event in trajectory.events if event.label == 'impulse']
    assert [round(event.time, 6) for event in events] == [2.0, 4.0, 6.0, 8.0, 10.0][: len(events)]
    assert len(events) >= 4
    assert all(event.after[rate] - event.before[rate] == pytest.approx(0.5) for event in events)


def test_circuit_and_oscillator_agree():
    """Test that the RLC circuit and the damped oscillator agree to the solver tolerance for equal coefficients."""
    charge = run('rlc_circuit', L=2.0, R=0.4, C=1 / 8, V=0.5, omega=1.3).trajectory_.values('circuit.q')
    displacement = run('driven_damped_oscillator', m=2.0, c=0.4, k=8.0, F=0.5, omega=1.3).trajectory_.values('mass.x')
    assert np.allclose(charge, displacement, rtol=1e-5, atol=1e-5)
