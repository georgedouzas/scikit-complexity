"""Test the `_replicate` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments._replicate import _replicate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
from skcomplex.systems.mathematics.stochastic_processes import WeakeningOrnsteinUhlenbeckSystem


def test_replicate_runs_many_initial_conditions_at_once():
    """Test that member i of a replicated logistic map follows the single map from the i-th initial value."""
    starts = np.linspace(0.05, 0.95, 100)
    replicated = _replicate(LogisticMapSystem(r=3.7).prepare().system_, 100, {'map.x': starts})
    values = np.asarray(simulate(replicated, until=50, dt=1).get_values('map.x'))
    for index in (0, 37, 99):
        single = simulate(LogisticMapSystem(x=float(starts[index]), r=3.7), until=50, dt=1)
        np.testing.assert_allclose(values[:, index], np.ravel(single.get_values('map.x')), atol=1e-12)


def test_replicate_runs_systems_with_interactions():
    """Test that member i of replicated predators and prey follows the single system from the i-th initial prey."""
    starts = np.asarray([20.0, 40.0, 60.0])
    replicated = _replicate(LotkaVolterraSystem().prepare().system_, 3, {'prey.x': starts})
    values = np.asarray(simulate(replicated, until=5, dt=0.1).get_values('predators.x'))
    for index, start in enumerate(starts):
        single = simulate(LotkaVolterraSystem(prey=float(start)), until=5, dt=0.1)
        np.testing.assert_allclose(values[:, index], np.ravel(single.get_values('predators.x')), rtol=1e-5)


def test_replicate_rejects_systems_with_groups():
    """Test that a system with a group does not replicate and the error says why."""
    with pytest.raises(ValidationError, match='Only systems of plain single elements and their interactions'):
        _replicate(WeakeningOrnsteinUhlenbeckSystem(4).prepare().system_, 3)


def test_replicate_checks_paths_and_rows():
    """Test that an unknown path and values without one row per member raise named errors."""
    with pytest.raises(ValidationError, match=r'map\.z is not a state or parameter'):
        _replicate(LogisticMapSystem().prepare().system_, 3, {'map.z': [1, 2, 3]})
    with pytest.raises(ValidationError, match='one row per member'):
        _replicate(LogisticMapSystem().prepare().system_, 3, {'map.r': [3.5, 3.6]})
