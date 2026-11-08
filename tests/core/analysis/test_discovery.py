"""Test the discovery of the analyses that apply to a trajectory, a system, or an array."""

import numpy as np

from skcomplex.core.analysis import ApplicableAnalysesResult, applicable_analyses, markov
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import PheromoneForagingSystem
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.mathematics.stochastic_processes import OnOffIntermittencySystem, PoissonProcessSystem
from skcomplex.systems.physics.active_matter import VicsekSystem
from skcomplex.systems.physics.statistical_mechanics import IsingSystem


def _discover(source: object) -> ApplicableAnalysesResult:
    """Return the analyses that apply to a source."""
    return applicable_analyses(source)


def _names(analyses: list) -> set[tuple[str, str | None]]:
    """Return the function name and the kind of every analysis."""
    return {(item.func.__name__, item.keywords.get('kind')) for item in analyses}


def test_discovery_of_a_lattice_system():
    """Test that an Ising system lists the structure factor, correlation length, clusters, and variance-mean law."""
    discovery = _discover(IsingSystem(size=16).prepare().system_)
    assert discovery.kinds == ['field']
    assert {
        ('spectrum', 'structure-factor'),
        ('correlation', 'length'),
        ('clusters', 'sizes'),
        ('scaling', 'variance-mean'),
    } <= _names(discovery.analyses['field'])
    assert 'agents' not in discovery.analyses
    assert discovery.readings['field'] == 'a grid of values, or frames of grids'


def test_discovery_of_a_system_with_several_kinds():
    """Test that an agent system on a lattice that records reporters lists agent, field, and series analyses."""
    discovery = _discover(simulate(PheromoneForagingSystem(), until=2, random_state=0))
    assert discovery.kinds == ['agents', 'field', 'series']
    assert ('spectrum', 'structure-factor') in _names(discovery.analyses['field'])
    assert ('spectrum', 'power') in _names(discovery.analyses['series'])


def test_discovery_of_an_agent_system():
    """Test that a Vicsek trajectory lists the alignment, nearest-neighbour, and displacement analyses under agents."""
    discovery = _discover(simulate(VicsekSystem(), until=3, random_state=0))
    agents = discovery.analyses['agents']
    assert {('alignment', 'mean-vector'), ('neighbors', 'distances'), ('displacement', None)} <= _names(agents)


def test_discovery_of_a_plain_array():
    """Test that a two-dimensional array is a field and a set of series read by column, and a vector one series."""
    discovery = _discover(np.zeros((16, 16)))
    assert discovery.kinds == ['field', 'series']
    assert discovery.readings == {
        'field': 'the array as one grid',
        'series': 'every column of the array as a series',
    }
    assert ('spectrum', 'structure-factor') in _names(discovery.analyses['field'])
    assert ('entropy', 'permutation') in _names(discovery.analyses['series'])
    assert _discover(np.zeros(5)).readings == {'series': 'the array as one series'}


def test_discovered_analyses_run_on_arrays_without_a_simulation():
    """Test that the entropies listed for a measured series run on the array alone."""
    values = np.random.default_rng(0).standard_normal(2000)
    entropies = [item for item in _discover(values).analyses['series'] if item.func.__name__ == 'entropy']
    assert {item.keywords['kind'] for item in entropies} >= {'shannon', 'permutation', 'sample', 'spectral'}
    permutation = next(item for item in entropies if item.keywords['kind'] == 'permutation')
    assert 0.9 < permutation(values).entropy[0] <= 1


def test_discovery_of_a_jump_path():
    """Test that a jump path lists Markov analyses, which run on its states."""
    trajectory = simulate(PoissonProcessSystem(n_processes=1), until=30.0, random_state=0)
    discovery = _discover(trajectory)
    assert 'path' in discovery.kinds
    assert any(item.func is markov for item in discovery.analyses['path'])
    states = np.asarray(trajectory.values(trajectory.problem.states.names[0]), dtype=int).ravel() % 3
    system = markov(np.tile(states, 20), kind='state-system')
    assert system.transition_matrix.shape[0] >= 2


def test_discovery_leaves_out_analyses_whose_tags_the_system_lacks():
    """Test that the Lyapunov exponents are listed for a deterministic system and left out for a stochastic one."""
    deterministic = applicable_analyses(simulate(LogisticGrowthSystem(), until=5.0))
    stochastic = applicable_analyses(simulate(OnOffIntermittencySystem(), until=5.0, random_state=0))
    assert ('lyapunov', 'largest') in _names(deterministic.analyses['series'])
    assert ('lyapunov', 'largest') not in _names(stochastic.analyses['series'])
    assert ('entropy', 'permutation') in _names(stochastic.analyses['series'])
