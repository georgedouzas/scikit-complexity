"""Test the `_utils` module of `skcomplex.core.laws`."""

import pytest

from skcomplex.core.base import symbol, vector
from skcomplex.core.laws import Call, EquationLaw, EventLaw, JumpLaw, NoiseLaw, RateLaw, Symbols, values_of


def test_symbols_unknown_name_lists_available_names():
    """Test that a missing symbol names the available ones."""
    with pytest.raises(AttributeError, match='Available names are: a'):
        _ = Symbols(a=1).b


def _double(t, value):
    """Return twice a value."""
    return 2 * value


def test_values_of_flattens_calls_vectors_and_mappings():
    """Test that calls give their arguments, vectors their components, and mappings their keys and values."""
    x, y, k = symbol('x'), symbol('y'), symbol('k')
    assert values_of(RateLaw(x, Call(_double, k * x))) == [x, k * x, None]
    assert values_of(RateLaw(vector([x, y]), vector([y, -x]))) == [x, y, y, -x, None]
    assert values_of(EventLaw(x, {y: 0, x: k})) == [x, y, 0, x, k, 0, False, None]


def test_values_of_reads_chosen_fields():
    """Test that only the included fields are read, without the excluded ones."""
    x, h = symbol('x'), symbol('h')
    noise = NoiseLaw(x, 0.1 * x, process='market')
    assert values_of(noise, include=('value',)) == [0.1 * x]
    assert 'market' not in values_of(noise, exclude=('process',))
    assert values_of(EquationLaw(x == h), include=('lhs', 'rhs')) == [x, h]
    assert values_of(JumpLaw(h * x, {x: -1}), include=('rate',)) == [h * x]
