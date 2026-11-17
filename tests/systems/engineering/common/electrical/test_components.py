"""Test the `_components` module of `skcomplex.systems.engineering.common.electrical`."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.signal import find_peaks

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.solvers import RK4Solver, ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.systems.engineering.circuits import ChuaSystem
from skcomplex.systems.engineering.common.electrical import (
    CapacitorInteraction,
    ChuaDiodeInteraction,
    CurrentSourceInteraction,
    DiodeInteraction,
    GroundInteraction,
    InductorInteraction,
    OperationalAmplifierInteraction,
    ResistorInteraction,
    VoltageSourceInteraction,
)


def _nodes(*names):
    """Return one node element of potential `v` per name."""
    return [Element(name, values={'v': 0.0}) for name in names]


def _series_rlc(solver='auto', backend='auto'):
    """Return a series RLC circuit that runs from a charged capacitor."""
    return System(
        elements=[
            *_nodes('top', 'middle', 'bottom'),
            Element('tank', values={'u': 1.0, 'i': 0.0}),
            Element('coil', values={'i': 0.0}),
            Element('earth', values={'i': 0.0}),
        ],
        interactions=[
            CapacitorInteraction('capacitor', a='top', b='bottom', branch='tank', C=0.2),
            ResistorInteraction('resistor', a='top', b='middle', R=0.05),
            InductorInteraction('inductor', a='middle', b='bottom', branch='coil', L=0.5),
            GroundInteraction('ground', node='bottom', branch='earth'),
        ],
        solver=solver,
        backend=backend,
    )


def _component_chua(solver='auto'):
    """Return Chua's circuit assembled from components with the values of `chua`."""
    return System(
        elements=[
            *_nodes('one', 'two', 'zero'),
            Element('first', values={'u': 0.1, 'i': 0.0}),
            Element('second', values={'u': 0.0, 'i': 0.0}),
            Element('coil', values={'i': 0.0}),
            Element('earth', values={'i': 0.0}),
        ],
        interactions=[
            CapacitorInteraction('first_capacitor', a='one', b='zero', branch='first', C=1 / 15.6),
            CapacitorInteraction('second_capacitor', a='two', b='zero', branch='second', C=1.0),
            InductorInteraction('inductor', a='zero', b='two', branch='coil', L=1 / 28.0),
            ResistorInteraction('resistor', a='one', b='two', R=1.0),
            ChuaDiodeInteraction('diode', a='one', b='zero'),
            GroundInteraction('ground', node='zero', branch='earth'),
        ],
        solver=solver,
    )


@pytest.mark.parametrize('solver', [ScipyODESolver(), RK4Solver(dt=1e-3)], ids=['scipy', 'rk4'])
def test_series_rlc_oscillates_at_its_resonance(solver):
    """Test that the capacitor voltage of a series RLC circuit oscillates within 1% of `1 / (2 pi sqrt(L C))`."""
    trajectory = _series_rlc(solver).prepare().simulate(until=10.0, dt=0.005)
    voltage = trajectory.get_values('tank.u').ravel()
    peaks, _ = find_peaks(voltage)
    frequency = (len(peaks) - 1) / (trajectory.times[peaks[-1]] - trajectory.times[peaks[0]])
    assert frequency == pytest.approx(1 / (2 * math.pi * math.sqrt(0.5 * 0.2)), rel=0.01)


def test_series_rlc_runs_on_the_numba_backend():
    """Test that the linear equations of a circuit lower to symbolic laws that the Numba backend runs."""
    trajectory = _series_rlc(backend=NumbaBackend()).prepare().simulate(until=1.0, dt=0.01)
    assert trajectory.problem.backend == 'numba'
    assert trajectory.get_values('tank.u')[-1] < 1.0


def test_currents_into_every_node_sum_to_zero():
    """Test that the currents of the capacitor, the resistor, and the inductor balance at the middle node."""
    trajectory = _series_rlc().prepare().simulate(until=2.0, dt=0.1)
    top, middle = trajectory.get_values('top.v'), trajectory.get_values('middle.v')
    np.testing.assert_allclose((top - middle) / 0.05, trajectory.get_values('coil.i'), atol=1e-8)
    np.testing.assert_allclose(trajectory.get_values('tank.i'), -trajectory.get_values('coil.i'), atol=1e-8)


def test_diode_current_solves_the_diode_equation():
    """Test that a resistor in series with an exponential diode carries the solution of the diode equation."""
    system = System(
        elements=[
            *_nodes('plus', 'middle', 'zero'),
            Element('cell', values={'i': 0.0}),
            Element('earth', values={'i': 0.0}),
        ],
        interactions=[
            VoltageSourceInteraction('source', a='plus', b='zero', branch='cell', V=1.0),
            ResistorInteraction('resistor', a='plus', b='middle', R=100.0),
            DiodeInteraction('diode', a='middle', b='zero', Is=1e-6, Vt=0.025),
            GroundInteraction('ground', node='zero', branch='earth'),
        ],
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.25)

    def balance(current):
        return 1e-6 * (math.exp((1.0 - 100.0 * current) / 0.025) - 1) - current

    expected = brentq(balance, 0.0, 0.01, xtol=1e-14)
    current = (trajectory.get_values('plus.v') - trajectory.get_values('middle.v')) / 100.0
    assert np.allclose(current.ravel(), expected, rtol=0, atol=1e-8)
    assert np.allclose(trajectory.get_values('cell.i').ravel(), -expected, rtol=0, atol=1e-8)


def test_current_source_drives_a_resistor():
    """Test that a current source sets the voltage of the resistor it drives."""
    system = System(
        elements=[*_nodes('top', 'zero'), Element('earth', values={'i': 0.0})],
        interactions=[
            CurrentSourceInteraction('source', a='top', b='zero', current=0.5),
            ResistorInteraction('load', a='top', b='zero', R=4.0),
            GroundInteraction('ground', node='zero', branch='earth'),
        ],
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.5)
    assert np.allclose(trajectory.get_values('top.v').ravel(), 2.0)


def test_operational_amplifier_inverts_with_the_ratio_of_its_resistors():
    """Test that an inverting amplifier has the gain `-R2 / R1`."""
    system = System(
        elements=[
            *_nodes('plus', 'sum', 'output', 'zero'),
            *[Element(name, values={'i': 0.0}) for name in ('cell', 'drive', 'earth')],
        ],
        interactions=[
            VoltageSourceInteraction('source', a='plus', b='zero', branch='cell', V=0.5),
            ResistorInteraction('input', a='plus', b='sum', R=1.0),
            ResistorInteraction('feedback', a='sum', b='output', R=3.0),
            OperationalAmplifierInteraction('amplifier', p='zero', n='sum', out='output', branch='drive'),
            GroundInteraction('ground', node='zero', branch='earth'),
        ],
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.5)
    assert np.allclose(trajectory.get_values('output.v').ravel(), -1.5)


def test_component_chua_follows_chua():
    """Test that the capacitor voltages of the component circuit follow `ChuaSystem` until chaos parts the runs."""
    solver = ScipyODESolver(method='RK45', rtol=1e-10, atol=1e-10)
    circuit = _component_chua(solver).prepare().simulate(until=20.0, dt=0.05)
    written = ChuaSystem(solver=solver).prepare().simulate(20.0, dt=0.05)
    assert np.allclose(
        circuit.get_values('first.u').ravel(),
        written.get_values('circuit.x').ravel(),
        rtol=0,
        atol=1e-4,
    )
    assert np.allclose(
        circuit.get_values('second.u').ravel(),
        written.get_values('circuit.y').ravel(),
        rtol=0,
        atol=1e-4,
    )


def test_component_chua_visits_both_scrolls():
    """Test that the component circuit meets the bounds of the test of `chua`."""
    trajectory = _component_chua().prepare().simulate(until=100.0, dt=0.05)
    voltage = trajectory.get_values('first.u')[400:]
    assert np.abs(voltage).max() < 3.0
    assert 0.2 < (voltage > 0).mean() < 0.8
