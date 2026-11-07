"""Test the signature analysis of every data kind."""

import numpy as np
import pytest

from skcomplex.core.analysis import signatures
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import create_simulation


def _signatures(trajectory: object) -> list:
    """Return the signatures of a trajectory."""
    return signatures(trajectory).signatures


def test_signature_analysis_of_a_series():
    """Test that a series gives its power spectrum on log axes with the autocorrelation time."""
    (signature,) = _signatures(create_simulation('lotka_volterra').prepare().run(until=50.0).trajectory_)
    assert signature.kind == 'series'
    assert signature.title == 'Power spectrum'
    assert (signature.xscale, signature.yscale) == ('log', 'log')
    assert set(signature.curves) == {'prey.n', 'predators.n'}
    assert any(name.startswith('autocorrelation time') for name in signature.values)


def test_signature_analysis_of_a_map():
    """Test that a map gives its return map as points."""
    (signature,) = _signatures(create_simulation('logistic_map').prepare().run(until=100).trajectory_)
    assert signature.points
    assert signature.title == 'Return map'


def test_signature_analysis_of_a_jump_path():
    """Test that a jump path gives its waiting-time distribution with the dispersion index."""
    trajectory = create_simulation('poisson_process').prepare().run(until=30.0).trajectory_
    (signature,) = _signatures(trajectory)
    assert signature.kind == 'path'
    assert 'dispersion index of jumps per frame' in signature.values
    ((_, survival),) = signature.curves.values()
    assert survival[0] == 1


def test_signature_analysis_of_a_field():
    """Test that a field gives the structure factor of its last frame with a length."""
    (signature,) = _signatures(create_simulation('ising', size=32).prepare().run(until=10).trajectory_)
    assert signature.kind == 'field'
    assert signature.ylabel == 'S(k)'
    assert {'correlation length', 'peak wavelength'} & set(signature.values)


def test_signature_analysis_of_agents():
    """Test that agents with velocities give their polar and milling orders over time."""
    trajectory = create_simulation('boids').prepare().run(until=5).trajectory_
    signature = _signatures(trajectory)[0]
    assert signature.kind == 'agents'
    assert set(signature.curves) == {'polar order', 'milling order'}
    assert signature.curves['polar order'][1].size == len(trajectory.times)


def test_signature_analysis_of_agents_without_motion():
    """Test that agents without velocities or headings give nearest-neighbour distances against the Poisson law."""
    signature = _signatures(create_simulation('diffusion_limited_aggregation').prepare().run(until=3).trajectory_)[0]
    assert signature.title in {'Nearest-neighbour distances', 'Agents present'}


def test_signature_analysis_of_a_network():
    """Test that a network gives its degree distribution with its moments."""
    signature = _signatures(create_simulation('network_diffusion').prepare().run(until=0.5).trajectory_)[0]
    assert signature.kind == 'network'
    assert signature.values['mean degree'] > 0
    assert 'clustering' in signature.values


def test_signature_analysis_of_group_states():
    """Test that the scalar states of a group without positions give a series signature with one curve per member."""
    (signature,) = _signatures(create_simulation('replicator_dynamics').prepare().run(until=5.0).trajectory_)
    assert signature.kind == 'series'
    assert set(signature.curves) == {'population.x[0]', 'population.x[1]'}


def test_signature_analysis_of_phases_of_particles():
    """Test that particles with a phase and no velocity give the order of their phases over time."""
    trajectory = create_simulation('swarmalators').prepare().run(until=1.0).trajectory_
    (signature,) = _signatures(trajectory)
    expected = np.abs(np.exp(1j * trajectory.values('units.theta')).mean(axis=1))
    assert signature.curves['polar order'][1] == pytest.approx(expected)


def test_signature_analysis_reads_a_trajectory():
    """Test that the signature analysis rejects a source that is not a trajectory."""
    with pytest.raises(AnalysisError, match='reads a trajectory'):
        signatures(np.zeros(10))


def test_signature_analysis_of_a_system_with_several_kinds():
    """Test that an agent system on a lattice with reporters gives agents, field, then series signatures."""
    trajectory = create_simulation('pheromone_foraging').prepare().run(until=5).trajectory_
    kinds = [signature.kind for signature in _signatures(trajectory)]
    assert kinds == ['agents', 'field', 'series']
