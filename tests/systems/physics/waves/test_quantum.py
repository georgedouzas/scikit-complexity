"""Test the `_quantum` module of `skcomplex.systems.physics.waves`."""

import numpy as np
import pytest

from skcomplex.core.analysis import OperatorAnalysis
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_system
from skcomplex.systems.physics.waves import schrodinger

from ..conftest import _relative_spread, _series


def test_schrodinger_packet_moves_at_the_group_velocity_and_keeps_its_norm():
    """Test that a free wave packet keeps its norm and its center moves at `2 D k`."""
    trajectory = Simulation(schrodinger(), dt=0.5).prepare().run(until=4.0).trajectory_
    u, w = _series(trajectory, 'packet.u', 'packet.w')
    density = u**2 + w**2
    positions = 40.0 / 256 * np.arange(256) - 20.0
    center = density @ positions / density.sum(1)
    assert _relative_spread(density.sum(1)) < 1e-6
    assert center[-1] - center[0] == pytest.approx(2 * 0.5 * 2.0 * 4.0, rel=1e-4)


def test_schrodinger_formulations_agree():
    """Test that the field and operator formulations give the same probability of every site."""
    spacing = 40.0 / 128
    fields = (
        Simulation(create_system('schrodinger', n_sites=128), dt=0.5, solver=ETDRK4Solver(dt=0.01))
        .prepare()
        .run(until=4.0)
    )
    operators = create_system('schrodinger', n_sites=128, formulation='operators')
    operators = Simulation(operators, compiler=OperatorCompiler(), dt=0.5).prepare().run(until=4.0)
    density = fields.trajectory_.values('packet.u') ** 2 + fields.trajectory_.values('packet.w') ** 2
    probability = np.abs(operators.trajectory_.values('schrodinger.state')) ** 2
    assert np.allclose(density * spacing, probability, atol=1e-6)


def test_schrodinger_dephased_packet_keeps_its_trace(tmp_path):
    """Test that a dephased packet keeps trace 1 while its purity falls, and streams its probability."""
    system = create_system('schrodinger', n_sites=32, length=20.0, formulation='operators', gamma=0.5)
    observer = RerunObserver('dephased', save_path=tmp_path / 'run.rrd')
    simulation = Simulation(system, compiler=OperatorCompiler(), dt=0.5, observers=[observer])
    trajectory = simulation.prepare().run(until=2.0).trajectory_
    traces = np.trace(trajectory.values('schrodinger.state'), axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)
    assert np.all(np.diff(OperatorAnalysis('purity').prepare(trajectory).run().purity_) <= 1e-10)
    with pytest.raises(ValidationError, match='formulation'):
        create_system('schrodinger', gamma=0.5)
