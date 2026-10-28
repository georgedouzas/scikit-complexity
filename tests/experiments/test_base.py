"""Test the `_base` module of `skcomplex.experiments`: the bases of designs and experiments, and the surface."""

import inspect

import numpy as np
import pytest
from scipy import stats

from skcomplex import experiments
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.experiments import BaseDesign, BaseExperiment, RandomDesign
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.mathematics.dynamical_systems import logistic_map


def test_surface_holds_only_classes():
    """Test that every public name of the package is a class and that the old functions are gone."""
    assert all(inspect.isclass(getattr(experiments, name)) for name in experiments.__all__)
    for name in ('calibrate', 'sensitivity', 'run_task', 'resolve_source', 'Task', 'RunResult', 'Source'):
        assert not hasattr(experiments, name), name


def test_bases_are_abstract():
    """Test that the bases of designs and experiments cannot be built without their hooks."""
    assert inspect.isabstract(BaseDesign)
    assert inspect.isabstract(BaseExperiment)


def test_sampled_design_reads_distributions():
    """Test that a sampled design maps its samples through the quantile function of a SciPy distribution."""
    rows = RandomDesign(n=200, random_state=0).check().check().prepare({'r': stats.norm(10, 1)}).run()
    assert 9.5 < sum(row['r'] for row in rows) / len(rows) < 10.5


@pytest.mark.parametrize('bound', [(0.0, 1.0, 2.0), 'low', 3.0], ids=['triple', 'text', 'number'])
def test_sampled_design_rejects_invalid_bounds(bound):
    """Test that a bound that is neither a pair nor a distribution raises an error naming it when prepared."""
    with pytest.raises(ValidationError, match='The bounds of r must be'):
        RandomDesign(n=4).check().check().prepare({'r': bound})


def test_design_check_rejects_negative_size():
    """Test that a negative number of rows builds and fails when checked."""
    with pytest.raises(ValidationError, match='nonnegative integer'):
        RandomDesign(n=-1).check()


def test_replicate_runs_many_initial_conditions_at_once():
    """Test that member i of a replicated logistic map follows the single map from the i-th initial value."""
    starts = np.linspace(0.05, 0.95, 100)
    ensemble = (
        Simulation(BaseExperiment._replicate(logistic_map(r=3.7), 100, {'map.x': starts}), dt=1)
        .check()
        .run(until=50)
    )
    values = np.asarray(ensemble.trajectory_.values('map.x'))
    for index in (0, 37, 99):
        single = Simulation(logistic_map(x=float(starts[index]), r=3.7), dt=1).check().run(until=50)
        np.testing.assert_allclose(values[:, index], np.ravel(single.trajectory_.values('map.x')), atol=1e-12)


def test_replicate_rejects_systems_with_interactions():
    """Test that a system with an interaction does not replicate and the error says why."""
    with pytest.raises(ValidationError, match='Only systems of plain single elements replicate'):
        BaseExperiment._replicate(lotka_volterra(), 3)


def test_replicate_checks_paths_and_rows():
    """Test that an unknown path and values without one row per member raise named errors."""
    with pytest.raises(ValidationError, match=r'map\.z is not a state or parameter'):
        BaseExperiment._replicate(logistic_map(), 3, {'map.z': [1, 2, 3]})
    with pytest.raises(ValidationError, match='one row per member'):
        BaseExperiment._replicate(logistic_map(), 3, {'map.r': [3.5, 3.6]})
