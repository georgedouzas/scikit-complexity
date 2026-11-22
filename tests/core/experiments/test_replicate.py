"""Test the `_replicate` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments._replicate import _replicate, _replicate_prepared
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
from skcomplex.systems.mathematics.stochastic_processes import WeakeningParticleElement


def test_replicate_runs_many_initial_conditions_at_once():
    """Test that member i of a replicated logistic map follows the single map from the i-th initial value."""
    starts = np.linspace(0.05, 0.95, 100)
    replicated = _replicate_prepared(LogisticMapSystem(r=3.7).prepare().system_, 100, {'map.x': starts})
    values = np.asarray(replicated.prepare().simulate(50, dt=1).get_values('map.x'))
    for index in (0, 37, 99):
        single = LogisticMapSystem(x=float(starts[index]), r=3.7).prepare().simulate(50, dt=1)
        np.testing.assert_allclose(values[:, index], np.ravel(single.get_values('map.x')), atol=1e-12)


def test_replicate_runs_systems_with_interactions():
    """Test that member i of replicated predators and prey follows the single system from the i-th initial prey."""
    starts = np.asarray([20.0, 40.0, 60.0])
    replicated = _replicate_prepared(LotkaVolterraSystem().prepare().system_, 3, {'prey.x': starts})
    values = np.asarray(replicated.prepare().simulate(until=5, dt=0.1).get_values('predators.x'))
    for index, start in enumerate(starts):
        single = LotkaVolterraSystem(prey=float(start)).prepare().simulate(until=5, dt=0.1)
        np.testing.assert_allclose(values[:, index], np.ravel(single.get_values('predators.x')), rtol=1e-5)


def test_replicate_rejects_systems_with_groups():
    """Test that a system with a group does not replicate and the error says why."""
    with pytest.raises(ValidationError, match='Only systems of single elements and their interactions'):
        _replicate(System(elements=[WeakeningParticleElement('particles', size=4)]), 3)


def test_replicate_checks_paths_and_rows():
    """Test that an unknown path and values without one row per member raise named errors."""
    with pytest.raises(ValidationError, match=r'map\.z is not a value of an element'):
        _replicate_prepared(LogisticMapSystem().prepare().system_, 3, {'map.z': [1, 2, 3]})
    with pytest.raises(ValidationError, match='one row per member'):
        _replicate_prepared(LogisticMapSystem().prepare().system_, 3, {'map.r': [3.5, 3.6]})


def test_replicate_prepared_rejects_elements_of_many_members():
    """Test that a system whose element has many members does not replicate and the error says why."""
    system = System(elements=[Element('map', values={'x': [0.1, 0.2], 'r': 3.7}, size=2)])
    with pytest.raises(ValidationError, match='Only systems of single elements and their interactions'):
        _replicate_prepared(system, 3)


def test_replicate_prepared_gives_every_member_its_own_parameter():
    """Test that member i of replicated logistic maps runs with the i-th growth rate."""
    rates = np.asarray([2.9, 3.2, 3.7])
    replicated = _replicate_prepared(LogisticMapSystem().prepare().system_, 3, {'map.r': rates})
    values = np.asarray(replicated.prepare().simulate(20, dt=1).get_values('map.x'))
    for index, rate in enumerate(rates):
        single = LogisticMapSystem(r=float(rate)).prepare().simulate(20, dt=1)
        np.testing.assert_allclose(values[:, index], np.ravel(single.get_values('map.x')), atol=1e-12)


def test_replicate_prepared_runs_elements_whose_class_takes_no_size():
    """Test that populations whose class takes no `size` replicate, member i starting from the i-th prey."""
    system = LotkaVolterraSystem().prepare().system_
    replicated = _replicate_prepared(system, 2, {'prey.x': [20.0, 40.0]}).prepare()
    values = np.asarray(replicated.simulate(1.0, dt=0.5).get_values('prey.x'))
    single = LotkaVolterraSystem(prey=40.0).prepare().simulate(1.0, dt=0.5)
    np.testing.assert_allclose(values[:, 1], np.ravel(single.get_values('prey.x')), rtol=1e-5)


def test_replicate_prepared_takes_one_starting_derivative_per_member():
    """Test that replicated masses that coast take their own starting velocity `mass.x'`."""
    coast = Interaction('coast', roles={'target': 'mass'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0)])
    system = System(elements=[Element('mass', values={'x': 0.0})], interactions=[coast])
    replicated = _replicate_prepared(system, 2, {"mass.x'": np.array([1.0, -2.0])}).prepare()
    final = replicated.simulate(until=1.0, dt=0.01).get_values('mass.x')[-1]
    assert final == pytest.approx(np.array([1.0, -2.0]), rel=1e-6)
