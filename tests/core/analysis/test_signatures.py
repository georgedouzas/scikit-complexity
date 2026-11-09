"""Test the signature analysis of every data kind."""

import numpy as np
import pytest

from skcomplex.core.analysis import signatures
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.collective_behaviour import BoidsSystem, PheromoneForagingSystem
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.biology.evolution import ReplicatorDynamicsSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
from skcomplex.systems.physics.statistical_mechanics import DiffusionLimitedAggregationSystem, IsingSystem
from skcomplex.systems.physics.synchronization import SwarmalatorsSystem


def _signatures(trajectory: object) -> list:
    """Return the signatures of a trajectory."""
    return signatures(trajectory).signatures


def test_signature_analysis_of_a_series():
    """Test that a series gives its power spectrum on log axes with the autocorrelation time."""
    (signature,) = _signatures(simulate(LotkaVolterraSystem(), until=50.0, random_state=0))
    assert signature.kind == 'series'
    assert signature.title == 'Power spectrum'
    assert (signature.xscale, signature.yscale) == ('log', 'log')
    assert set(signature.curves) == {'prey.x', 'predators.x'}
    assert any(name.startswith('autocorrelation time') for name in signature.values)


def test_signature_analysis_of_a_map():
    """Test that a map gives its return map as points."""
    (signature,) = _signatures(simulate(LogisticMapSystem(), until=100, random_state=0))
    assert signature.points
    assert signature.title == 'Return map'


def test_signature_analysis_of_a_jump_path():
    """Test that a jump path gives its waiting-time distribution with the dispersion index."""
    trajectory = simulate(PoissonProcessSystem(), until=30.0, random_state=0)
    (signature,) = _signatures(trajectory)
    assert signature.kind == 'path'
    assert 'dispersion index of jumps per frame' in signature.values
    ((_, survival),) = signature.curves.values()
    assert survival[0] == 1


def test_signature_analysis_of_a_field():
    """Test that a field gives the structure factor of its last frame with a length."""
    (signature,) = _signatures(simulate(IsingSystem(size=32), until=10, random_state=0))
    assert signature.kind == 'field'
    assert signature.ylabel == 'S(k)'
    assert {'correlation length', 'peak wavelength'} & set(signature.values)


def test_signature_analysis_of_agents():
    """Test that agents with velocities give their polar and milling orders over time."""
    trajectory = simulate(BoidsSystem(), until=5, random_state=0)
    signature = _signatures(trajectory)[0]
    assert signature.kind == 'agents'
    assert set(signature.curves) == {'polar order', 'milling order'}
    assert signature.curves['polar order'][1].size == len(trajectory.times)


def test_signature_analysis_of_agents_without_motion():
    """Test that agents without velocities or headings give nearest-neighbour distances against the Poisson law."""
    signature = _signatures(simulate(DiffusionLimitedAggregationSystem(), until=3, random_state=0))[0]
    assert signature.title in {'Nearest-neighbour distances', 'Agents present'}


def test_signature_analysis_of_a_network():
    """Test that a network gives its degree distribution with its moments."""
    signature = _signatures(simulate(NetworkDiffusionSystem(), until=0.5, random_state=0))[0]
    assert signature.kind == 'network'
    assert signature.values['mean degree'] > 0
    assert 'clustering' in signature.values


def test_signature_analysis_of_group_states():
    """Test that the scalar states of a group without positions give a series signature with one curve per member."""
    (signature,) = _signatures(simulate(ReplicatorDynamicsSystem(), until=5.0, random_state=0))
    assert signature.kind == 'series'
    assert set(signature.curves) == {'population.x[0]', 'population.x[1]'}


def test_signature_analysis_of_phases_of_particles():
    """Test that particles with a phase and no velocity give the order of their phases over time."""
    trajectory = simulate(SwarmalatorsSystem(), until=1.0, random_state=0)
    (signature,) = _signatures(trajectory)
    expected = np.abs(np.exp(1j * trajectory.values('units.theta')).mean(axis=1))
    assert signature.curves['polar order'][1] == pytest.approx(expected)


def test_signature_analysis_reads_a_trajectory():
    """Test that the signature analysis rejects a source that is not a trajectory."""
    with pytest.raises(AnalysisError, match='reads a trajectory'):
        signatures(np.zeros(10))


def test_signature_analysis_of_a_system_with_several_kinds():
    """Test that an agent system on a lattice with reporters gives agents, field, then series signatures."""
    trajectory = simulate(PheromoneForagingSystem(), until=5, random_state=0)
    kinds = [signature.kind for signature in _signatures(trajectory)]
    assert kinds == ['agents', 'field', 'series']
