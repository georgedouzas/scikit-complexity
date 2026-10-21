"""Test the replication of systems of single elements into vectorised ensembles."""

import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.model import replicate
from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.mathematics.dynamical_systems import logistic_map


def test_replicate_runs_many_initial_conditions_at_once():
    """Test that member i of a replicated logistic map follows the single map from the i-th initial value."""
    starts = np.linspace(0.05, 0.95, 100)
    ensemble = Simulation(replicate(logistic_map(r=3.7), 100, {'map.x': starts}), dt=1).run(until=50)
    values = np.asarray(ensemble.trajectory_['map.x'])
    for index in (0, 37, 99):
        single = Simulation(logistic_map(x=float(starts[index]), r=3.7), dt=1).run(until=50)
        np.testing.assert_allclose(values[:, index], np.ravel(single.trajectory_['map.x']), atol=1e-12)


def test_replicate_rejects_systems_with_interactions():
    """Test that a system with an interaction does not replicate and the error says why."""
    with pytest.raises(ValidationError, match='Only systems of plain single elements replicate'):
        replicate(lotka_volterra(), 3)


def test_replicate_checks_paths_and_rows():
    """Test that an unknown path and values without one row per member raise named errors."""
    with pytest.raises(ValidationError, match=r'map\.z is not a state or parameter'):
        replicate(logistic_map(), 3, {'map.z': [1, 2, 3]})
    with pytest.raises(ValidationError, match='one row per member'):
        replicate(logistic_map(), 3, {'map.r': [3.5, 3.6]})
