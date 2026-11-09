"""Test the `_draws` module of `skcomplex.core.components.laws.utils`."""

import importlib
from multiprocessing.reduction import ForkingPickler

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws.utils import bernoulli, categorical, draw, normal, uniform
from skcomplex.core.components.laws.utils._draws import draws_in
from skcomplex.core.errors import ValidationError


def test_uniform_merges_uses_of_one_name():
    """Test that two uses of one draw name are one symbolic value."""
    assert str(uniform('u', 0, 1) - uniform('u', 0, 1)) == '0'


def test_uniform_keeps_different_names_apart():
    """Test that draws with different names stay two values."""
    assert str(uniform('u') - uniform('v')) == 'uniform_u(0, 1) - uniform_v(0, 1)'


def test_normal_rejects_names_that_are_not_identifiers():
    """Test that a draw name must be a Python identifier."""
    with pytest.raises(ValidationError, match='identifier'):
        normal('two words')


def test_categorical_rejects_empty_weights():
    """Test that a categorical draw needs at least one weight."""
    with pytest.raises(ValidationError, match='nonempty'):
        categorical('k', [])


def test_draws_in_finds_nested_draws():
    """Test that draws in the arguments of other draws are found, and other expressions hold none."""
    x = var('x')
    found = draws_in(x + bernoulli('flip', uniform('p')))
    assert sorted(item.name for item in found) == ['flip', 'p']
    assert draws_in(x + 1) == []


def test_draw_of_a_scipy_distribution_keeps_its_keywords():
    """Test that a draw from a `scipy.stats` distribution keeps its name and its arguments by keyword."""
    (found,) = draws_in(draw('wait', 'gamma', a=2.0, scale=var('s')))
    assert (found.kind, found.name, found.distribution) == ('draw', 'wait', 'gamma')
    assert found.parameters == {'a': 2.0, 'scale': var('s')}


def test_draw_pickles_with_its_name():
    """Test that a draw survives pickling with its kind and name."""
    restored = ForkingPickler.loads(ForkingPickler.dumps(categorical('k', [1, 2])))
    (found,) = draws_in(restored)
    assert (found.kind, found.name) == ('categorical', 'k')


def test_draw_has_no_derivative():
    """Test that a draw differentiates to zero."""
    x = var('x')
    assert uniform('u', 0, x).diff(x) == 0


def test_relaxed_draws_keep_their_relaxation():
    """Test that a relaxed draw keeps its temperature and hard flag, and differs from the exact draw."""
    (found,) = draws_in(bernoulli('flip', 0.5, temperature=0.2, hard=True))
    assert (found.temperature, found.hard) == (0.2, True)
    assert str(bernoulli('flip', 0.5) - bernoulli('flip', 0.5, temperature=0.2)) != '0'


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'temperature': 0.0}, 'positive temperature'), ({'hard': True}, 'only with a temperature')],
)
def test_relaxed_draws_reject_invalid_relaxations(arguments, match):
    """Test that a relaxation needs a positive temperature, and `hard` needs a temperature."""
    with pytest.raises(ValidationError, match=match):
        categorical('k', [1, 2], **arguments)


def test_draw_functions_have_one_import_path():
    """Test that the law namespace defines the draw functions and no other public core package exports them."""
    law_utils = importlib.import_module('skcomplex.core.components.laws.utils')
    specs = importlib.import_module('skcomplex.core.specs')
    assert law_utils.uniform.__module__ == 'skcomplex.core.components.laws.utils._draws'
    assert not any(hasattr(specs, name) for name in ('uniform', 'normal', 'bernoulli', 'categorical', 'draw'))
