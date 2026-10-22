"""Test the data kinds, the trajectory adapters, and the discovery of applicable analyses."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import (
    ANALYSES,
    AnalysisEntry,
    DataKind,
    analyses_for,
    analysis,
    data_kinds,
    ensemble_of,
    fields_of,
    markov_state_model,
    networks_of,
    path_of,
    series_of,
    snapshots_of,
    structure_factor,
)
from skcomplex.core.base import AnalysisError, Domain, parameter, state
from skcomplex.core.experiments import Experiment
from skcomplex.core.model import Element, Jump, System
from skcomplex.core.runtime import Simulation
from skcomplex.models import create, run

BirthDeath = Element.define(
    'BirthDeath',
    states={'n': state(default=10.0, domain=Domain.NONNEGATIVE_INTEGER)},
    parameters={'birth': parameter(5.0), 'death': parameter(0.5)},
    laws=lambda s: [Jump(s.birth, {s.n: 1}), Jump(s.death * s.n, {s.n: -1})],
)


def test_structure_factor_same_function_on_two_domains():
    """Test that an Ising frame and a Gray-Scott frame give the same structure factor type."""
    spins = fields_of(run('ising', size=32, settings={'until': 5}).trajectory_)[-1]
    gray_scott = fields_of(run('gray_scott', settings={'until': 40.0}).trajectory_)[-1]
    first, second = structure_factor(spins), structure_factor(gray_scott)
    assert type(first) is type(second)
    assert first.power.ndim == second.power.ndim == 1


def test_analyses_for_arrays_without_a_simulation():
    """Test that every listed analysis of a measured series runs on the array alone."""
    values = np.random.default_rng(0).standard_normal(2000)
    found = {item.name for item in analyses_for(values)}
    assert {'autocorrelation', 'sample_entropy', 'hurst_exponent'} <= found
    assert ANALYSES.resolve('sample_entropy').function(values) > 1


def test_series_of_an_ode():
    """Test that the series of the prey and predators equal the trajectory's columns."""
    trajectory = run('lotka_volterra', settings={'until': 5.0}).trajectory_
    series = series_of(trajectory, ['prey.n', 'predators.n'])
    assert series.values.shape == (len(trajectory), 2)
    np.testing.assert_array_equal(series.values[:, 0], trajectory['prey.n'])
    np.testing.assert_array_equal(series.values[:, 1], trajectory['predators.n'])
    np.testing.assert_array_equal(series.times, trajectory.times)


def test_path_of_a_gillespie_run():
    """Test that the jump path of a birth-death run keeps only the frames where the state changed."""
    system = System(components=[BirthDeath('cells')])
    trajectory = Simulation(system, dt=0.01, random_state=0).run(until=20).trajectory_
    path = path_of(trajectory)
    assert path.label == 'cells.n'
    assert np.all(np.diff(path.states) != 0)
    assert np.all(np.abs(np.diff(path.states)) >= 1)
    assert path.times.size < len(trajectory)


def test_fields_of_a_lattice():
    """Test that the frames of a 64 by 64 Ising run of 100 frames have the lattice shape."""
    trajectory = run('ising', size=64, settings={'until': 99}).trajectory_
    frames = fields_of(trajectory, 'spins.spin')
    assert frames.shape == (100, 64, 64)
    np.testing.assert_array_equal(frames, trajectory.lattice('spins.spin'))


def test_snapshots_of_unwraps_agent_positions():
    """Test that walkers in a periodic square of side 10 are unwrapped so no step exceeds half the side."""
    trajectory = run('run_and_tumble', n_agents=50, speed=1.0, settings={'until': 200}).trajectory_
    snapshots = snapshots_of(trajectory)
    steps = np.linalg.norm(np.diff(snapshots.positions, axis=0), axis=-1)
    assert snapshots.periodic
    assert steps.max() < 5
    assert np.ptp(snapshots.positions) > 10
    assert snapshots.headings.shape == (201, 50)


def test_snapshots_of_particle_groups():
    """Test that a particle group gives its positions and velocities in its periodic box."""
    trajectory = run('ideal_gas', settings={'until': 0.2}).trajectory_
    snapshots = snapshots_of(trajectory)
    assert snapshots.positions.shape == snapshots.velocities.shape
    assert snapshots.positions.shape[0] == len(trajectory)


def test_snapshots_of_missing_data_kind():
    """Test that an ODE trajectory has no agent group and the error lists its data kinds."""
    trajectory = run('lotka_volterra', settings={'until': 1.0}).trajectory_
    with pytest.raises(AnalysisError, match=r'no agent group.*series'):
        snapshots_of(trajectory)


def test_networks_of_a_network_space():
    """Test that a field on a network gives its graph at every frame."""
    trajectory = run('network_diffusion', settings={'until': 0.5}).trajectory_
    graphs = networks_of(trajectory)
    assert len(graphs) == len(trajectory)
    assert nx.is_isomorphic(graphs[0], nx.karate_club_graph())


def test_ensemble_of_replicates():
    """Test that 200 replicates of an Ornstein-Uhlenbeck process give one row per run with its seed."""
    result = Experiment('hasselmann', replicates=200, until=2.0, random_state=0).run()
    name = next(iter(result.runs[0].series))
    ensemble = ensemble_of(result, name)
    assert ensemble.values.shape == (200, ensemble.times.size)
    assert len(set(ensemble.seeds.tolist())) == 200
    assert ensemble.values[:, -1].std() > 0


def test_analyses_for_lattice_model():
    """Test that an Ising system lists field analyses and no agent analysis."""
    found = analyses_for(create('ising', size=16))
    fields = {item.name for item in found if item.kind is DataKind.FIELD}
    assert {'structure_factor', 'correlation_length', 'cluster_sizes', 'binder_cumulant'} <= fields
    assert all(item.kind is not DataKind.AGENTS for item in found)
    assert all(item.adapter == 'fields_of' for item in found if item.kind is DataKind.FIELD)


def test_analyses_for_agent_model():
    """Test that a flocking trajectory lists polarization, neighbour statistics, and displacement as agent analyses."""
    trajectory = run('boids', settings={'until': 3}).trajectory_
    agents = {item.name for item in analyses_for(trajectory) if item.kind is DataKind.AGENTS}
    assert {'polar_order', 'nearest_neighbor_distances', 'mean_squared_displacement'} <= agents


def test_analyses_for_plain_array():
    """Test that a two-dimensional array lists field analyses and series analyses with their readings."""
    found = analyses_for(np.zeros((16, 16)))
    readings = {(item.kind, item.reading) for item in found}
    assert (DataKind.FIELD, 'the array as one grid') in readings
    assert (DataKind.SERIES, 'every column of the array as a series') in readings
    assert data_kinds(np.zeros(5)) == [DataKind.SERIES]


def test_analyses_for_jump_path_in_the_default_install():
    """Test that a jump path lists the Markov state model, which runs in the default install."""
    trajectory = run('poisson_process', n_processes=1, settings={'until': 30.0, 'random_state': 0}).trajectory_
    names = {item.name for item in analyses_for(trajectory) if item.kind is DataKind.PATH}
    assert 'markov_state_model' in names
    path = path_of(trajectory)
    states = np.asarray(path.states, dtype=int) % 3
    assert markov_state_model(np.tile(states, 20)).transition_matrix.shape[0] >= 2


def test_analysis_registers_a_new_analysis():
    """Test that the decorator registers a user's analysis under its data kind and returns it unchanged."""

    def spread(values):
        """Return the range of values."""
        return float(np.ptp(values))

    registered = analysis(DataKind.SERIES)(spread)
    assert registered is spread
    assert ANALYSES.resolve('spread') == AnalysisEntry('spread', spread, DataKind.SERIES)
    assert DataKind.SERIES.adapter == 'series_of'
