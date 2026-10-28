"""Test the `_base` module of `skcomplex.core.laws`."""

import inspect

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var, vector

from skcomplex.core import laws
from skcomplex.core.laws import BaseLaw, CrossingLaw, JumpLaw, LocalRuleLaw, RateLaw
from skcomplex.core.spaces import BoxNeighborhood
from skcomplex.errors import NotCheckedError, ValidationError
from skcomplex.specs import CallSpec, RequirementSpec, TotalisticSpec


def test_every_law_derives_from_the_law_base():
    """Test that every law class of the package derives from the law base and implements its hooks."""
    classes = [getattr(laws, name) for name in laws.__all__ if name.endswith('Law') and name != 'BaseLaw']
    assert len(classes) > 20
    for law in classes:
        assert issubclass(law, BaseLaw), law
        assert not inspect.isabstract(law), law


def test_law_runs_alone_once_prepared():
    """Test that a prepared rate law knows its targets, reads, and owner, and runs into its rate at given values."""
    n, r = var('n r')
    law = RateLaw(n, r * n).check().prepare('meadow/rabbits')
    assert law.run({'n': 10, 'r': 0.5}) == 5.0
    assert law.targets_ == (n,)
    assert law.reads_ == {n, r}
    assert law.owner_ == 'meadow/rabbits'


def test_law_without_an_owner_has_none():
    """Test that a law prepared without the path of its owner has no owner."""
    assert RateLaw(var('x'), 1).check().prepare().owner_ is None


def test_law_must_be_checked_before_it_is_prepared():
    """Test that preparing a law that was not checked raises an error."""
    with pytest.raises(NotCheckedError):
        RateLaw(var('x'), 1).prepare()


def test_law_contributes_itself():
    """Test that a law contributes itself as its one law, whatever the namespace."""
    law = RateLaw(var('x'), 1)
    assert law._get_laws(None) == [law]


def test_targets_are_a_tuple_once_prepared():
    """Test that preparing a law sets its targets as a tuple, and that laws without targets have none."""
    x, y = var('x y')
    assert RateLaw(x, 1).check().prepare().targets_ == (x,)
    assert LocalRuleLaw(y, BoxNeighborhood(), TotalisticSpec({2: 1})).check().prepare().targets_ == (y,)
    assert CrossingLaw(x - 1).check().prepare().targets_ == ()


def test_law_reads_the_symbols_of_its_nested_arguments():
    """Test that the reads of a law include the symbols inside vectors, sequences, and mappings."""
    x, y, a, b = var('x y a b')
    law = RateLaw(vector([x, y]), vector([a * x, b])).check().prepare()
    assert law.reads_ == {x, y, a, b}


def test_law_requirements_are_a_requirement_spec():
    """Test that a rate law requires continuous time."""
    requirements = RateLaw(var('x'), 1)._get_requirements()
    assert isinstance(requirements, RequirementSpec)
    assert [time.__name__ for time in requirements.time] == ['ContinuousTime']


def test_law_rejects_a_numpy_array_value():
    """Test that a NumPy array, which NumPy builds from a scalar times a vector, is rejected with a fix."""
    with pytest.raises(ValidationError, match='NumPy array'):
        RateLaw(vector([var('x')]), np.float64(2.0) * vector([var('x')])).check()


def _double(t, value):
    """Return twice a value."""
    return 2 * value


def test_values_flatten_calls_vectors_and_mappings():
    """Test that the values of a law give the arguments of calls, the components of vectors, and keys and values."""
    x, y, k = var('x y k')
    assert RateLaw(x, CallSpec(_double, (k * x,)))._get_values() == [x, k * x, None]
    assert RateLaw(vector([x, y]), vector([y, -x]))._get_values() == [x, y, y, -x, None]
    assert JumpLaw(k * x, {x: -1, y: 1})._get_values() == [k * x, x, -1, y, 1, None, None]
