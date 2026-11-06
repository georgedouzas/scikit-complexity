"""Test the discovery of the analyses that apply to a trajectory, a system, or an array."""

import numpy as np

from skcomplex.core.analysis import (
    AlignmentAnalysis,
    DiscoveryAnalysis,
    DisplacementAnalysis,
    EntropyAnalysis,
    MarkovAnalysis,
    NeighborAnalysis,
)
from skcomplex.systems import create_simulation, create_system


def _discover(source: object) -> DiscoveryAnalysis:
    """Return the discovery of a source, run."""
    return DiscoveryAnalysis().prepare(source).run()


def _names(analyses: list) -> set[tuple[str, str | None]]:
    """Return the class name and the kind of every analysis."""
    return {(type(item).__name__, getattr(item, 'kind', None)) for item in analyses}


def test_discovery_of_a_lattice_system():
    """Test that an Ising system lists the structure factor, correlation length, clusters, and variance-mean law."""
    discovery = _discover(create_system('ising', size=16))
    assert discovery.kinds_ == ['field']
    assert {
        ('SpectrumAnalysis', 'structure-factor'),
        ('CorrelationAnalysis', 'length'),
        ('ClusterAnalysis', 'sizes'),
        ('ScalingAnalysis', 'variance-mean'),
    } <= _names(discovery.analyses_['field'])
    assert 'agents' not in discovery.analyses_
    assert discovery.readings_['field'] == 'a grid of values, or frames of grids'


def test_discovery_of_a_system_with_several_kinds():
    """Test that an agent system on a lattice that records reporters lists agent, field, and series analyses."""
    discovery = _discover(create_simulation('pheromone_foraging').prepare().run(until=2).trajectory_)
    assert discovery.kinds_ == ['agents', 'field', 'series']
    assert ('SpectrumAnalysis', 'structure-factor') in _names(discovery.analyses_['field'])
    assert ('SpectrumAnalysis', 'power') in _names(discovery.analyses_['series'])


def test_discovery_of_an_agent_system():
    """Test that a Vicsek trajectory lists the alignment, nearest-neighbour, and displacement analyses under agents."""
    discovery = _discover(create_simulation('vicsek').prepare().run(until=3).trajectory_)
    agents = discovery.analyses_['agents']
    assert any(isinstance(item, AlignmentAnalysis) and item.kind == 'mean-vector' for item in agents)
    assert any(isinstance(item, NeighborAnalysis) and item.kind == 'distances' for item in agents)
    assert any(isinstance(item, DisplacementAnalysis) for item in agents)


def test_discovery_of_a_plain_array():
    """Test that a two-dimensional array is a field and a set of series read by column, and a vector one series."""
    discovery = _discover(np.zeros((16, 16)))
    assert discovery.kinds_ == ['field', 'series']
    assert discovery.readings_ == {
        'field': 'the array as one grid',
        'series': 'every column of the array as a series',
    }
    assert ('SpectrumAnalysis', 'structure-factor') in _names(discovery.analyses_['field'])
    assert ('EntropyAnalysis', 'permutation') in _names(discovery.analyses_['series'])
    assert _discover(np.zeros(5)).readings_ == {'series': 'the array as one series'}


def test_discovered_analyses_run_on_arrays_without_a_simulation():
    """Test that the entropy analyses listed for a measured series run on the array alone."""
    values = np.random.default_rng(0).standard_normal(2000)
    entropies = [item for item in _discover(values).analyses_['series'] if isinstance(item, EntropyAnalysis)]
    assert {item.kind for item in entropies} >= {'shannon', 'permutation', 'sample', 'spectral'}
    permutation = next(item for item in entropies if item.kind == 'permutation')
    assert 0.9 < permutation.prepare(values).run().entropy_[0] <= 1


def test_discovery_of_a_jump_path():
    """Test that a jump path lists Markov analyses, which run on its states."""
    trajectory = create_simulation('poisson_process', n_processes=1).prepare().run(until=30.0).trajectory_
    discovery = _discover(trajectory)
    assert 'path' in discovery.kinds_
    assert any(isinstance(item, MarkovAnalysis) for item in discovery.analyses_['path'])
    states = np.asarray(trajectory.values(trajectory.problem.states.names[0]), dtype=int).ravel() % 3
    system = MarkovAnalysis('state-system').prepare(np.tile(states, 20)).run()
    assert system.transition_matrix_.shape[0] >= 2
