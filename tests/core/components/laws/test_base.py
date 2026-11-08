"""Test the `_base` module of `skcomplex.core.components.laws`."""

import inspect

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var, vector

from skcomplex.core.components import laws
from skcomplex.core.components.laws import BaseLaw, CrossingLaw, EquationLaw, JumpLaw, LocalRuleLaw, NoiseLaw, RateLaw
from skcomplex.core.components.spaces import BoxNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import CallSpec, RequirementSpec, TotalisticSpec


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
    law = RateLaw(n, r * n).prepare('meadow/rabbits')
    assert law.run({'n': 10, 'r': 0.5}).value_ == 5.0
    assert law.targets_ == (n,)
    assert law.reads_ == {n, r}
    assert law.owner_ == 'meadow/rabbits'


def test_law_without_an_owner_has_none():
    """Test that a law prepared without the path of its owner has no owner."""
    assert RateLaw(var('x'), 1).prepare().owner_ is None


def test_law_contributes_itself():
    """Test that a law contributes itself as its one law, whatever the namespace."""
    law = RateLaw(var('x'), 1)
    assert law._get_laws(None) == [law]


def test_targets_are_a_tuple_once_prepared():
    """Test that preparing a law sets its targets as a tuple, and that laws without targets have none."""
    x, y = var('x y')
    assert RateLaw(x, 1).prepare().targets_ == (x,)
    assert LocalRuleLaw(y, BoxNeighborhood(), TotalisticSpec({2: 1})).prepare().targets_ == (y,)
    assert CrossingLaw(x - 1).prepare().targets_ == ()


def test_law_reads_the_symbols_of_its_nested_arguments():
    """Test that the reads of a law include the symbols inside vectors, sequences, and mappings."""
    x, y, a, b = var('x y a b')
    law = RateLaw(vector([x, y]), vector([a * x, b])).prepare()
    assert law.reads_ == {x, y, a, b}


def test_law_requirements_are_a_requirement_spec():
    """Test that a rate law requires continuous time."""
    requirements = RateLaw(var('x'), 1)._get_requirements()
    assert isinstance(requirements, RequirementSpec)
    assert [time.__name__ for time in requirements.time] == ['ContinuousTime']


def test_law_rejects_a_numpy_array_value():
    """Test that a NumPy array, which NumPy builds from a scalar times a vector, is rejected with a fix."""
    with pytest.raises(ValidationError, match='NumPy array'):
        RateLaw(vector([var('x')]), np.float64(2.0) * vector([var('x')])).prepare()


def _double(t, value):
    """Return twice a value."""
    return 2 * value


def test_values_flatten_calls_vectors_and_mappings():
    """Test that the values of a law give the arguments of calls, the components of vectors, and keys and values."""
    x, y, k = var('x y k')
    assert RateLaw(x, CallSpec(_double, (k * x,)))._get_values() == [x, k * x, None, None]
    assert RateLaw(vector([x, y]), vector([y, -x]))._get_values() == [x, y, y, -x, None, None]
    assert JumpLaw(k * x, {x: -1, y: 1})._get_values() == [k * x, x, -1, y, 1, None, None]


def test_get_requirements_reads_every_law_class_without_an_instance():
    """Test that every public law class states its time and space before any law of it exists."""
    classes = [getattr(laws, name) for name in laws.__all__]
    for law in [
        item for item in classes if inspect.isclass(item) and issubclass(item, BaseLaw) and item is not BaseLaw
    ]:
        assert isinstance(law.get_requirements(), RequirementSpec)
    assert JumpLaw.get_requirements().time == RateLaw.get_requirements().time


def test_laws_give_their_kind_times_and_randomness():
    """Test that a law gives its kind, the times it runs in, and a randomness that follows from its kind."""
    x = var('x')
    rate = RateLaw(x, -x).get_tags()
    assert (rate.laws, rate.time, rate.randomness) == ({'rate'}, {'continuous'}, 'deterministic')
    assert NoiseLaw(x, 1.0).get_tags().randomness == 'stochastic'
    assert EquationLaw(x).get_tags().time == {'continuous', 'static'}


def test_every_core_law_names_a_known_kind():
    """Test that every law of the core names one of the law kinds of the tags."""
    for law in [item for item in vars(laws).values() if inspect.isclass(item) and issubclass(item, BaseLaw)]:
        if law is not BaseLaw:
            assert law.law_tag in {'rate', 'noise', 'jump', 'update', 'equation', 'sampling'}, law.__name__
