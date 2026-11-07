"""Test the `_quantum_spins` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np
import pytest
import rerun.blueprint as rrb
from scipy.special import jv

from skcomplex.core.analysis import operator_measures
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import LindbladSolver
from skcomplex.core.engine.viz import ExpectationVisual, RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_simulation, create_system


def test_quantum_heisenberg_ground_state_of_the_ring():
    """Test that the ring of four spins has the ground energy -2."""
    system = create_system('quantum_heisenberg', n_sites=4, J=1.0, delta=1.0, h=0.0)
    energies = operator_measures(system, n_states=1).energies
    assert energies[0] == pytest.approx(-2.0, abs=1e-12)


def test_quantum_heisenberg_single_magnon_follows_bessel_functions():
    """Test that one flipped spin of the XX ring spreads as J_d(J t)² with a conserved magnetisation."""
    trajectory = (
        create_simulation('quantum_heisenberg', delta=0.0).set_params(dt=0.1).prepare().run(until=2.0).trajectory_
    )
    sz = trajectory.values('sz')
    for distance in range(-5, 6):
        expected = jv(abs(distance), trajectory.times) ** 2
        assert np.allclose(0.5 - sz[:, 6 + distance], expected, atol=1e-5)
    assert np.ptp(trajectory.values('magnetization')) < 1e-8


def test_quantum_heisenberg_run_by_name_records_states_and_observables():
    """Test that a run by name records 21 complex states and the local magnetisation of every site."""
    trajectory = create_simulation('quantum_heisenberg').set_params(dt=0.1).prepare().run(until=2.0).trajectory_
    assert len(trajectory.times) == 21
    assert trajectory.values('quantum_heisenberg.state').shape == (21, 4096)
    assert np.iscomplexobj(trajectory.values('quantum_heisenberg.state'))
    assert trajectory.values('sz').shape == (21, 12)


def test_quantum_heisenberg_dephasing_destroys_coherence():
    """Test that dephasing uses the Lindblad solver, lowers the purity, and conserves the magnetisation."""
    simulation = (
        create_simulation('quantum_heisenberg', n_sites=6, initial='neel', gamma=0.1)
        .set_params(dt=0.2)
        .prepare()
        .run(until=4.0)
    )
    assert isinstance(simulation.stepper_, LindbladSolver)
    values = operator_measures(simulation.trajectory_, kind='purity').purity
    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < 1
    assert np.ptp(simulation.trajectory_.values('magnetization')) < 1e-8


@pytest.mark.parametrize(('values', 'message'), [({'boundary': 'twisted'}, 'boundary'), ({'initial': 'x'}, 'x')])
def test_quantum_heisenberg_rejects_unknown_options(values, message):
    """Test that unknown boundaries and initial states raise an error."""
    with pytest.raises(ValidationError, match=message):
        create_system('quantum_heisenberg', **values)


def test_quantum_heisenberg_streams_a_light_cone(tmp_path):
    """Test that the chain logs a space-time image of the local magnetisation as its primary view."""
    system = create_system('quantum_heisenberg', initial='domain_wall', boundary='open')
    blueprint = RerunObserver(blueprint='default').prepare(OperatorCompiler().prepare(system).run().problem_).blueprint_
    first = blueprint.root_container.contents[0]
    assert isinstance(first, rrb.Container | rrb.Spatial2DView)
    observer = RerunObserver('light_cone', save_path=tmp_path / 'run.rrd')
    Simulation(system, dt=0.5, observers=[observer], compiler=OperatorCompiler()).prepare().run(until=2.0)
    assert observer.logged_frames_ == 5
    assert repr(ExpectationVisual()) == 'ExpectationVisual()'
