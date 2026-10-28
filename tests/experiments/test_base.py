"""Test the `_base` module of `skcomplex.experiments`: the bases of designs and experiments, and the surface."""

import inspect

import pytest
from scipy import stats

from skcomplex import experiments
from skcomplex.core.errors import ValidationError
from skcomplex.experiments import BaseDesign, BaseExperiment, RandomDesign


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
    rows = RandomDesign(n=200, random_state=0).check().prepare({'r': stats.norm(10, 1)}).run()
    assert 9.5 < sum(row['r'] for row in rows) / len(rows) < 10.5


@pytest.mark.parametrize('bound', [(0.0, 1.0, 2.0), 'low', 3.0], ids=['triple', 'text', 'number'])
def test_sampled_design_rejects_invalid_bounds(bound):
    """Test that a bound that is neither a pair nor a distribution raises an error naming it when prepared."""
    with pytest.raises(ValidationError, match='The bounds of r must be'):
        RandomDesign(n=4).check().prepare({'r': bound})


def test_design_check_rejects_negative_size():
    """Test that a negative number of rows builds and fails when checked."""
    with pytest.raises(ValidationError, match='nonnegative integer'):
        RandomDesign(n=-1).check()
