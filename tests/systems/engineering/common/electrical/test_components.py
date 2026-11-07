"""Test the `_components` module of `skcomplex.systems.engineering.common.electrical`."""

import math
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.signal import find_peaks

from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver, ScipyODESolver
from skcomplex.core.specs import ConnectorSpec
from skcomplex.systems.engineering.circuits import ChuaSystem
from skcomplex.systems.engineering.common.electrical import (
    ELECTRICAL,
    CapacitorElement,
    ChuaDiodeElement,
    CurrentSourceElement,
    DiodeElement,
    GroundElement,
    InductorElement,
    OperationalAmplifierElement,
    ResistorElement,
    VoltageSourceElement,
)


def _nodes(*nodes):
    """Return one connection per node, each a list of pairs of an element and a port."""
    return [
        ConnectionInteraction(
            f'node{position}',
            members=[member for member, _ in node],
            ports=[port for _, port in node],
        )
        for position, node in enumerate(nodes)
    ]


def _series_rlc(R=0.05, L=0.5, C=0.2, u=1.0):
    """Return a series RLC circuit that runs from a charged capacitor."""
    return System(
        elements=[
            ResistorElement('resistor', R=R),
            InductorElement('coil', L=L),
            CapacitorElement('tank', C=C, u=u),
            GroundElement('ground'),
        ],
        interactions=_nodes(
            [('tank', 'p'), ('resistor', 'p')],
            [('resistor', 'n'), ('coil', 'p')],
            [('coil', 'n'), ('tank', 'n'), ('ground', 'p')],
        ),
    )


def _component_chua():
    """Return Chua's circuit assembled from components with the values of `chua`."""
    return System(
        elements=[
            CapacitorElement('first', C=1 / 15.6, u=0.1),
            CapacitorElement('second', C=1.0, u=0.0),
            InductorElement('coil', L=1 / 28.0, i=0.0),
            ResistorElement('resistor', R=1.0),
            ChuaDiodeElement('diode'),
            GroundElement('ground'),
        ],
        interactions=_nodes(
            [('first', 'p'), ('resistor', 'p'), ('diode', 'p')],
            [('second', 'p'), ('resistor', 'n'), ('coil', 'n')],
            [('first', 'n'), ('second', 'n'), ('coil', 'p'), ('diode', 'n'), ('ground', 'p')],
        ),
    )


def test_electrical_connector_lives_in_the_library():
    """Test that the connector is a `ConnectorSpec` that no module of the core or the engine names."""
    assert isinstance(ELECTRICAL, ConnectorSpec)
    for package in ('skcomplex.core.components', 'skcomplex.core.engine'):
        for module in Path(str(files(package))).rglob('*.py'):
            assert "'electrical'" not in module.read_text()


@pytest.mark.parametrize('solver', [ScipyODESolver(), RK4Solver(dt=1e-3)], ids=['scipy', 'rk4'])
def test_series_rlc_oscillates_at_its_resonance(solver):
    """Test that the capacitor voltage of a series RLC circuit oscillates within 1% of `1 / (2 pi sqrt(L C))`."""
    trajectory = Simulation(_series_rlc(), dt=0.005, solver=solver).prepare().run(until=10.0).trajectory_
    voltage = trajectory.values('tank.u').ravel()
    peaks, _ = find_peaks(voltage)
    frequency = (len(peaks) - 1) / (trajectory.times[peaks[-1]] - trajectory.times[peaks[0]])
    assert frequency == pytest.approx(1 / (2 * math.pi * math.sqrt(0.5 * 0.2)), rel=0.01)


def test_series_rlc_runs_on_the_numba_backend():
    """Test that the linear equations of a circuit lower to symbolic laws that the Numba backend runs."""
    simulation = Simulation(_series_rlc(), dt=0.01, backend=NumbaBackend()).prepare().run(until=1.0)
    assert simulation.problem_.backend == 'numba'
    assert simulation.trajectory_.values('tank.u')[-1] < 1.0


def test_diode_current_solves_the_diode_equation():
    """Test that a resistor in series with an exponential diode carries the solution of the diode equation."""
    system = System(
        elements=[
            VoltageSourceElement('source', V=1.0),
            ResistorElement('resistor', R=100.0),
            DiodeElement('diode', Is=1e-6, Vt=0.025),
            GroundElement('ground'),
        ],
        interactions=_nodes(
            [('source', 'p'), ('resistor', 'p')],
            [('resistor', 'n'), ('diode', 'p')],
            [('diode', 'n'), ('source', 'n'), ('ground', 'p')],
        ),
    )
    trajectory = Simulation(system, dt=0.25).prepare().run(until=1.0).trajectory_

    def balance(current):
        return 1e-6 * (math.exp((1.0 - 100.0 * current) / 0.025) - 1) - current

    expected = brentq(balance, 0.0, 0.01, xtol=1e-14)
    assert np.allclose(trajectory.values('diode.p_i').ravel(), expected, rtol=0, atol=1e-8)


def test_current_source_drives_a_resistor():
    """Test that a current source sets the voltage of the resistor it drives."""
    system = System(
        elements=[CurrentSourceElement('source', current=0.5), ResistorElement('load', R=4.0), GroundElement('ground')],
        interactions=_nodes(
            [('source', 'p'), ('load', 'p')],
            [('source', 'n'), ('load', 'n'), ('ground', 'p')],
        ),
    )
    trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
    assert np.allclose(trajectory.values('load.p_v').ravel(), 2.0)


def test_operational_amplifier_inverts_with_the_ratio_of_its_resistors():
    """Test that an inverting amplifier has the gain `-R2 / R1`."""
    system = System(
        elements=[
            VoltageSourceElement('source', V=0.5),
            ResistorElement('input', R=1.0),
            ResistorElement('feedback', R=3.0),
            OperationalAmplifierElement('amplifier'),
            GroundElement('ground'),
        ],
        interactions=_nodes(
            [('source', 'p'), ('input', 'p')],
            [('input', 'n'), ('feedback', 'p'), ('amplifier', 'n')],
            [('feedback', 'n'), ('amplifier', 'out')],
            [('source', 'n'), ('amplifier', 'p'), ('ground', 'p')],
        ),
    )
    trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
    assert np.allclose(trajectory.values('amplifier.out_v').ravel(), -1.5)


def test_component_chua_follows_chua():
    """Test that the capacitor voltages of the component circuit follow `ChuaSystem` until chaos separates the two runs."""
    settings = {'dt': 0.05, 'solver': ScipyODESolver(method='RK45', rtol=1e-10, atol=1e-10)}
    circuit = Simulation(_component_chua(), **settings).prepare().run(until=20.0).trajectory_
    written = Simulation(ChuaSystem().prepare().system_, **settings).prepare().run(until=20.0).trajectory_
    assert np.allclose(circuit.values('first.u').ravel(), written.values('circuit.x').ravel(), rtol=0, atol=1e-4)
    assert np.allclose(circuit.values('second.u').ravel(), written.values('circuit.y').ravel(), rtol=0, atol=1e-4)


def test_component_chua_visits_both_scrolls():
    """Test that the component circuit meets the bounds of the test of `chua`."""
    trajectory = Simulation(_component_chua(), dt=0.05).prepare().run(until=100.0).trajectory_
    voltage = trajectory.values('first.u')[400:]
    assert np.abs(voltage).max() < 3.0
    assert 0.2 < (voltage > 0).mean() < 0.8
