"""Test the `_circuits` module of `skcomplex.systems.engineering.circuits`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ParameterError
from skcomplex.systems.engineering.circuits import (
    ChuaSystem,
    ImpulsiveCircuitsSystem,
    MemristorOscillatorSystem,
    MixedModeChuaSystem,
    RlcCircuitSystem,
    TunnelDiodeOscillatorSystem,
)
from skcomplex.systems.physics.mechanics import DrivenDampedOscillatorSystem


def _peaks(series):
    """Return the values of the local maxima of a series."""
    inner = series[1:-1]
    return inner[(inner > series[:-2]) & (inner > series[2:])]


def test_chua_double_scroll_visits_both_scrolls():
    """Test that the double-scroll attractor stays bounded and visits both sides of the origin."""
    trajectory = simulate(ChuaSystem().prepare().system_, until=100, dt=0.05)
    x = trajectory.get_values('circuit.x')[400:]
    assert np.abs(x).max() < 3.0
    assert 0.2 < (x > 0).mean() < 0.8


def test_chua_sine_diode_has_several_scrolls():
    """Test that the sine diode gives a bounded attractor spanning more than two scrolls."""
    trajectory = simulate(ChuaSystem(nonlinearity='sine').prepare().system_, until=600, dt=0.5)
    x = trajectory.get_values('circuit.x')[100:]
    assert np.abs(x).max() < 15.0
    assert x.min() < -4 * 1.3
    assert x.max() > 4 * 1.3


def test_chua_rejects_unknown_nonlinearity():
    """Test that an unknown diode characteristic raises a parameter error."""
    with pytest.raises(ParameterError, match='Unknown nonlinearity'):
        ChuaSystem(nonlinearity='cubic').prepare()


@pytest.mark.parametrize(('c', 'mixed'), [(0.00053, True), (0.00105, False)])
def test_mixed_mode_chua_oscillations(c, mixed):
    """Test that small oscillations come between the spikes for `c = 0.00053` and not for `c = 0.00105`."""
    solver = ScipyODESolver(method='Radau')
    trajectory = simulate(MixedModeChuaSystem(c=c).prepare().system_, until=50, dt=0.1, solver=solver)
    peaks = _peaks(trajectory.get_values('circuit.x')[100:])
    assert (peaks > 1.0).sum() >= 4
    assert ((peaks < 0.3).sum() >= 4) == mixed


def test_memristor_oscillator_keeps_oscillating():
    """Test that the memristor circuit neither decays nor diverges."""
    trajectory = simulate(MemristorOscillatorSystem().prepare().system_, until=200, dt=0.1)
    voltage = trajectory.get_values('circuit.v')
    assert np.abs(voltage).max() < 10.0
    assert np.ptp(voltage[-500:]) > 0.5


def test_tunnel_diode_limit_cycle_amplitude():
    """Test that the tunnel-diode oscillator settles on the van der Pol limit cycle of amplitude `2 sqrt(alpha / 3)`."""
    trajectory = simulate(TunnelDiodeOscillatorSystem(alpha=1.0, gamma=1.0).prepare().system_, until=50, dt=0.05)
    assert np.abs(trajectory.get_values('circuit.V')[-300:]).max() == pytest.approx(2 * math.sqrt(1 / 3), rel=0.02)


def test_impulsive_circuit_kicks_the_rate():
    """Test that every impulse fires once per period and raises the rate by `A / (L C)`."""
    system = ImpulsiveCircuitsSystem(A=0.5, L=2.0, C=0.5, T=2.0).prepare().system_
    rate = PassCompiler().prepare(system).run().problem_.states.names.index('circuit.x_d1')
    trajectory = simulate(system, until=10, dt=0.05)
    events = [event for event in trajectory.events if event.label == 'impulse']
    assert [round(event.time, 6) for event in events] == [2.0, 4.0, 6.0, 8.0, 10.0][: len(events)]
    assert len(events) >= 4
    assert all(event.after[rate] - event.before[rate] == pytest.approx(0.5) for event in events)


def test_circuit_and_oscillator_agree():
    """Test that the RLC circuit and the damped oscillator agree to the solver tolerance for equal coefficients."""
    charge = simulate(RlcCircuitSystem(L=2.0, R=0.4, C=1 / 8, V=0.5, omega=1.3)).get_values('circuit.q')
    displacement = (
        simulate(DrivenDampedOscillatorSystem(m=2.0, c=0.4, k=8.0, F=0.5, omega=1.3), random_state=0)
        .get_values('mass.x')
        .ravel()
    )
    assert np.allclose(charge, displacement, rtol=1e-5, atol=1e-5)
