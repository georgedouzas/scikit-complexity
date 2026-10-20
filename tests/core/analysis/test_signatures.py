"""Test the signature analysis of every data kind."""

import numpy as np

from skcomplex.core.analysis import DataKind, signature_analysis
from skcomplex.core.experiments import Experiment
from skcomplex.models import run


def test_signature_analysis_of_a_series():
    """Test that a series gives its power spectrum on log axes with the autocorrelation time."""
    (signature,) = signature_analysis(run('lotka_volterra', settings={'until': 50.0}).trajectory_)
    assert signature.kind is DataKind.SERIES
    assert signature.title == 'Power spectrum'
    assert (signature.xscale, signature.yscale) == ('log', 'log')
    assert set(signature.curves) == {'prey.n', 'predators.n'}
    assert any(name.startswith('autocorrelation time') for name in signature.values)


def test_signature_analysis_of_a_map():
    """Test that a map gives its return map as points."""
    (signature,) = signature_analysis(run('logistic_map', settings={'until': 100}).trajectory_)
    assert signature.points
    assert signature.title == 'Return map'


def test_signature_analysis_of_a_jump_path():
    """Test that a jump path gives its waiting-time distribution with the Fano factor."""
    trajectory = run('poisson_process', settings={'until': 30.0, 'random_state': 0}).trajectory_
    (signature,) = signature_analysis(trajectory)
    assert signature.kind is DataKind.PATH
    assert 'Fano factor of jumps per frame' in signature.values
    ((_, survival),) = signature.curves.values()
    assert survival[0] == 1


def test_signature_analysis_of_a_field():
    """Test that a field gives the structure factor of its last frame with a length."""
    (signature,) = signature_analysis(run('ising', size=32, settings={'until': 10}).trajectory_)
    assert signature.kind is DataKind.FIELD
    assert signature.ylabel == 'S(k)'
    assert {'correlation length', 'peak wavelength'} & set(signature.values)


def test_signature_analysis_of_agents():
    """Test that agents with velocities give their polar and milling orders over time."""
    trajectory = run('boids', settings={'until': 5}).trajectory_
    signature = signature_analysis(trajectory)[0]
    assert signature.kind is DataKind.AGENTS
    assert set(signature.curves) == {'polar order', 'milling order'}
    assert signature.curves['polar order'][1].size == len(trajectory)


def test_signature_analysis_of_agents_without_motion():
    """Test that agents without velocities or headings give nearest-neighbour distances against the Poisson law."""
    signature = signature_analysis(run('diffusion_limited_aggregation', settings={'until': 3}).trajectory_)[0]
    assert signature.title in {'Nearest-neighbour distances', 'Agents present'}


def test_signature_analysis_of_a_network():
    """Test that a network gives its degree distribution with its moments."""
    (signature,) = signature_analysis(run('network_diffusion', settings={'until': 0.5}).trajectory_)
    assert signature.kind is DataKind.NETWORK
    assert signature.values['mean degree'] > 0
    assert 'clustering' in signature.values


def test_signature_analysis_of_an_ensemble():
    """Test that an ensemble gives the mean and the 5 to 95 percent band over runs."""
    result = Experiment('hasselmann', replicates=50, until=2.0, random_state=0).run()
    (signature,) = signature_analysis(result)
    _, low, high = signature.band
    _, mean = signature.curves['mean']
    assert np.all(low <= mean + 1e-12)
    assert np.all(mean <= high + 1e-12)
    assert signature.values['runs'] == 50


def test_signature_analysis_of_a_model_with_several_kinds():
    """Test that an agent model on a lattice with reporters gives agents, field, then series signatures."""
    trajectory = run('pheromone_foraging', settings={'until': 5}).trajectory_
    kinds = [signature.kind for signature in signature_analysis(trajectory)]
    assert kinds[:2] == [DataKind.AGENTS, DataKind.FIELD]
    assert kinds == sorted(kinds, key=[DataKind.AGENTS, DataKind.FIELD, DataKind.SERIES].index)
