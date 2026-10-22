"""Test the named random draws of update laws."""

import pickle  # nosec B403: the tests pickle and load only objects they create

import pytest

from skcomplex.core.base import ValidationError, symbol
from skcomplex.core.laws import Draw, bernoulli, categorical, draws_in, is_draw, normal, uniform


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
    """Test that draws in the arguments of other draws are found."""
    x = symbol('x')
    found = draws_in(x + bernoulli('flip', uniform('p')))
    assert sorted(draw.name for draw in found) == ['flip', 'p']


def test_draw_of_rejects_other_expressions():
    """Test that only draw calls convert to draws."""
    with pytest.raises(ValidationError, match='not a random draw'):
        Draw.of(symbol('x') + 1)


def test_is_draw_pickles_with_its_name():
    """Test that a draw survives pickling with its kind and name."""
    data = pickle.dumps(categorical('k', [1, 2]))
    restored = pickle.loads(data)  # noqa: S301  # nosec B301: loads only what it dumped
    assert is_draw(restored)
    assert (Draw.of(restored).kind, Draw.of(restored).name) == ('categorical', 'k')


def test_draws_in_has_no_derivative():
    """Test that a draw differentiates to zero."""
    x = symbol('x')
    assert uniform('u', 0, x).diff(x) == 0
