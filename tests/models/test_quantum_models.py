"""Test the quantum models: the Heisenberg chain and the operator formulation of the Schrödinger model."""

import numpy as np
import pytest
import rerun.blueprint as rrb
from scipy.special import jv

from skcomplex.core.analysis import purity, spectrum
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ETDRK4, LindbladEvolution
from skcomplex.core.viz import ExpectationVisual, RerunObserver, default_blueprint
from skcomplex.models import create, run


def test_quantum_heisenberg_ground_state_of_the_ring():
    """Test that the ring of four spins has the ground energy -2."""
    energies = spectrum(create('quantum_heisenberg', n_sites=4, J=1.0, delta=1.0, h=0.0), k=1).energies
    assert energies[0] == pytest.approx(-2.0, abs=1e-12)


def test_quantum_heisenberg_single_magnon_follows_bessel_functions():
    """Test that one flipped spin of the XX ring spreads as J_d(J t)² with a conserved magnetisation."""
    trajectory = run('quantum_heisenberg', delta=0.0, settings={'until': 2.0, 'dt': 0.1}).trajectory_
    sz = trajectory['sz']
    for distance in range(-5, 6):
        expected = jv(abs(distance), trajectory.times) ** 2
        assert np.allclose(0.5 - sz[:, 6 + distance], expected, atol=1e-5)
    assert np.ptp(trajectory['magnetization']) < 1e-8


def test_quantum_heisenberg_run_by_name_records_states_and_observables():
    """Test that a run by name records 21 complex states and the local magnetisation of every site."""
    trajectory = run('quantum_heisenberg', settings={'until': 2.0, 'dt': 0.1}).trajectory_
    assert len(trajectory.times) == 21
    assert trajectory['quantum_heisenberg.state'].shape == (21, 4096)
    assert np.iscomplexobj(trajectory['quantum_heisenberg.state'])
    assert trajectory['sz'].shape == (21, 12)


def test_quantum_heisenberg_dephasing_destroys_coherence():
    """Test that dephasing uses the Lindblad solver, lowers the purity, and conserves the magnetisation."""
    simulation = run('quantum_heisenberg', n_sites=6, initial='neel', gamma=0.1, settings={'until': 4.0, 'dt': 0.2})
    assert isinstance(simulation.stepper_, LindbladEvolution)
    values = purity(simulation.trajectory_)
    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < 1
    assert np.ptp(simulation.trajectory_['magnetization']) < 1e-8


@pytest.mark.parametrize(('values', 'message'), [({'boundary': 'twisted'}, 'boundary'), ({'initial': 'x'}, 'x')])
def test_quantum_heisenberg_rejects_unknown_options(values, message):
    """Test that unknown boundaries and initial states raise an error."""
    with pytest.raises(ValidationError, match=message):
        create('quantum_heisenberg', **values)


def test_quantum_heisenberg_streams_a_light_cone(tmp_path):
    """Test that the chain logs a space-time image of the local magnetisation as its primary view."""
    system = create('quantum_heisenberg', initial='domain_wall', boundary='open')
    blueprint = default_blueprint(system)
    first = blueprint.root_container.contents[0]
    assert isinstance(first, rrb.Container | rrb.Spatial2DView)
    observer = RerunObserver('light_cone', save_path=tmp_path / 'run.rrd')
    Simulation(system, dt=0.5, observers=[observer]).run(until=2.0)
    assert observer.logged_frames_ == 5
    assert repr(ExpectationVisual()) == 'ExpectationVisual(states=None, low=None, high=None, history=None)'


def test_schrodinger_formulations_agree():
    """Test that the field and operator formulations give the same probability of every site."""
    spacing = 40.0 / 128
    fields = Simulation(create('schrodinger', n_sites=128), dt=0.5, solver=ETDRK4(dt=0.01)).run(until=4.0)
    operators = Simulation(create('schrodinger', n_sites=128, formulation='operators'), dt=0.5).run(until=4.0)
    density = fields.trajectory_['packet.u'] ** 2 + fields.trajectory_['packet.w'] ** 2
    probability = np.abs(operators.trajectory_['schrodinger.state']) ** 2
    assert np.allclose(density * spacing, probability, atol=1e-6)


def test_schrodinger_dephased_packet_keeps_its_trace(tmp_path):
    """Test that a dephased packet keeps trace 1 while its purity falls, and streams its probability."""
    system = create('schrodinger', n_sites=32, length=20.0, formulation='operators', gamma=0.5)
    observer = RerunObserver('dephased', save_path=tmp_path / 'run.rrd')
    trajectory = Simulation(system, dt=0.5, observers=[observer]).run(until=2.0).trajectory_
    traces = np.trace(trajectory['schrodinger.state'], axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)
    assert np.all(np.diff(purity(trajectory)) <= 1e-10)
    with pytest.raises(ValidationError, match='formulation'):
        create('schrodinger', gamma=0.5)
