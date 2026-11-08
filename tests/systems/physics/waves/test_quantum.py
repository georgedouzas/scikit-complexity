"""Test the `_quantum` module of `skcomplex.systems.physics.waves`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import operator_measures
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.waves import SchrodingerSystem
from skcomplex.systems.physics.waves._quantum import SiteElement, site_space

from ..conftest import _relative_spread, _series


def test_schrodinger_packet_moves_at_the_group_velocity_and_keeps_its_norm():
    """Test that a free wave packet keeps its norm and its center moves at `2 D k`."""
    trajectory = simulate(SchrodingerSystem(), dt=0.5)
    u, w = _series(trajectory, 'packet.u', 'packet.w')
    density = u**2 + w**2
    positions = 40.0 / 256 * np.arange(256) - 20.0
    center = density @ positions / density.sum(1)
    assert _relative_spread(density.sum(1)) < 1e-6
    assert center[-1] - center[0] == pytest.approx(2 * 0.5 * 2.0 * 4.0, rel=1e-4)


def test_schrodinger_formulations_agree():
    """Test that the field and operator formulations give the same probability of every site."""
    spacing = 40.0 / 128
    fields = simulate(SchrodingerSystem(n_sites=128), dt=0.5, solver=ETDRK4Solver(dt=0.01))
    operators = simulate(SchrodingerSystem(n_sites=128, formulation='operators'), dt=0.5)
    density = fields.values('packet.u') ** 2 + fields.values('packet.w') ** 2
    probability = np.abs(operators.values('system.state')) ** 2
    assert np.allclose(density * spacing, probability, atol=1e-6)


def test_schrodinger_dephased_packet_keeps_its_trace(tmp_path):
    """Test that a dephased packet keeps trace 1 while its purity falls, and streams its probability."""
    system = SchrodingerSystem(n_sites=32, length=20.0, formulation='operators', gamma=0.5)
    observer = RerunObserver('dephased', save_path=tmp_path / 'run.rrd')
    trajectory = simulate(system, until=2.0, dt=0.5, observers=[observer])
    traces = np.trace(trajectory.values('system.state'), axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)
    assert np.all(np.diff(operator_measures(trajectory, kind='purity').purity) <= 1e-10)
    with pytest.raises(ValidationError, match='formulation'):
        SchrodingerSystem(gamma=0.5).prepare()


def test_site_operators_hold_projectors_and_the_adjacency():
    """Test that a particle on sites has projectors, the adjacency, and the Laplacian of its network."""
    site = SiteElement('p', hilbert=site_space(NetworkSpace(nx.cycle_graph(4)), projectors=True))
    operators = OperatorCompiler._view_of(site).local_operators()
    assert operators['site_2'].matrix.toarray()[2, 2] == 1
    assert operators['adjacency'].matrix.sum() == 8
    assert operators['laplacian'].matrix.toarray()[0, 0] == -2
