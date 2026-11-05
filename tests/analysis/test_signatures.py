"""Test the signature analysis of every data kind."""

import numpy as np
import pytest

from skcomplex.analysis import SignatureAnalysis
from skcomplex.errors import AnalysisError
from skcomplex.models import run


def _signatures(trajectory: object) -> list:
    """Return the signatures of a trajectory."""
    return SignatureAnalysis().prepare(trajectory).run().signatures_


def test_signature_analysis_of_a_series():
    """Test that a series gives its power spectrum on log axes with the autocorrelation time."""
    (signature,) = _signatures(run('lotka_volterra', settings={'until': 50.0}).trajectory_)
    assert signature.kind == 'series'
    assert signature.title == 'Power spectrum'
    assert (signature.xscale, signature.yscale) == ('log', 'log')
    assert set(signature.curves) == {'prey.n', 'predators.n'}
    assert any(name.startswith('autocorrelation time') for name in signature.values)


def test_signature_analysis_of_a_map():
    """Test that a map gives its return map as points."""
    (signature,) = _signatures(run('logistic_map', settings={'until': 100}).trajectory_)
    assert signature.points
    assert signature.title == 'Return map'


def test_signature_analysis_of_a_jump_path():
    """Test that a jump path gives its waiting-time distribution with the dispersion index."""
    trajectory = run('poisson_process', settings={'until': 30.0, 'random_state': 0}).trajectory_
    (signature,) = _signatures(trajectory)
    assert signature.kind == 'path'
    assert 'dispersion index of jumps per frame' in signature.values
    ((_, survival),) = signature.curves.values()
    assert survival[0] == 1


def test_signature_analysis_of_a_field():
    """Test that a field gives the structure factor of its last frame with a length."""
    (signature,) = _signatures(run('ising', size=32, settings={'until': 10}).trajectory_)
    assert signature.kind == 'field'
    assert signature.ylabel == 'S(k)'
    assert {'correlation length', 'peak wavelength'} & set(signature.values)


def test_signature_analysis_of_agents():
    """Test that agents with velocities give their polar and milling orders over time."""
    trajectory = run('boids', settings={'until': 5}).trajectory_
    signature = _signatures(trajectory)[0]
    assert signature.kind == 'agents'
    assert set(signature.curves) == {'polar order', 'milling order'}
    assert signature.curves['polar order'][1].size == len(trajectory.times)


def test_signature_analysis_of_agents_without_motion():
    """Test that agents without velocities or headings give nearest-neighbour distances against the Poisson law."""
    signature = _signatures(run('diffusion_limited_aggregation', settings={'until': 3}).trajectory_)[0]
    assert signature.title in {'Nearest-neighbour distances', 'Agents present'}


def test_signature_analysis_of_a_network():
    """Test that a network gives its degree distribution with its moments."""
    signature = _signatures(run('network_diffusion', settings={'until': 0.5}).trajectory_)[0]
    assert signature.kind == 'network'
    assert signature.values['mean degree'] > 0
    assert 'clustering' in signature.values


def test_signature_analysis_of_group_states():
    """Test that the scalar states of a group without positions give a series signature with one curve per member."""
    (signature,) = _signatures(run('replicator_dynamics', settings={'until': 5.0}).trajectory_)
    assert signature.kind == 'series'
    assert set(signature.curves) == {'population.x[0]', 'population.x[1]'}


def test_signature_analysis_of_phases_of_particles():
    """Test that particles with a phase and no velocity give the order of their phases over time."""
    trajectory = run('swarmalators', settings={'until': 1.0}).trajectory_
    (signature,) = _signatures(trajectory)
    expected = np.abs(np.exp(1j * trajectory.values('units.theta')).mean(axis=1))
    assert signature.curves['polar order'][1] == pytest.approx(expected)


def test_signature_analysis_reads_a_trajectory():
    """Test that the signature analysis rejects a source that is not a trajectory."""
    with pytest.raises(AnalysisError, match='reads a trajectory'):
        SignatureAnalysis().prepare(np.zeros(10))


def test_signature_analysis_of_a_model_with_several_kinds():
    """Test that an agent model on a lattice with reporters gives agents, field, then series signatures."""
    trajectory = run('pheromone_foraging', settings={'until': 5}).trajectory_
    kinds = [signature.kind for signature in _signatures(trajectory)]
    assert kinds == ['agents', 'field', 'series']
