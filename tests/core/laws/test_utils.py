"""Test the `_utils` module of `skcomplex.core.laws`."""

import traceback
from types import SimpleNamespace

import pytest

from skcomplex.core.laws import CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw
from skcomplex.core.laws.utils import Call
from skcomplex.engine.compile import values_of
from skcomplex.engine.problems import symbol, vector


def test_symbols_unknown_name_raises_attribute_error():
    """Test that a missing symbol raises Python's own error, which names the close symbol."""
    with pytest.raises(AttributeError, match="has no attribute 'rabit'") as error:
        _ = SimpleNamespace(rabbit=1).rabit
    assert "Did you mean: 'rabbit'?" in ''.join(traceback.format_exception_only(error.value))


def _double(t, value):
    """Return twice a value."""
    return 2 * value


def test_values_of_flattens_calls_vectors_and_mappings():
    """Test that calls give their arguments, vectors their components, and mappings their keys and values."""
    x, y, k = symbol('x'), symbol('y'), symbol('k')
    assert values_of(RateLaw(x, Call(_double, (k * x,)))) == [x, k * x, None]
    assert values_of(RateLaw(vector([x, y]), vector([y, -x]))) == [x, y, y, -x, None]
    assert values_of(CrossingLaw(x, {y: 0, x: k})) == [x, y, 0, x, k, 0, False]


def test_values_of_reads_chosen_fields():
    """Test that only the included fields are read, without the excluded ones."""
    x, h = symbol('x'), symbol('h')
    noise = NoiseLaw(x, 0.1 * x, process='market')
    assert values_of(noise, include=('value',)) == [0.1 * x]
    assert 'market' not in values_of(noise, exclude=('process',))
    assert values_of(EquationLaw(x == h).check(), include=('lhs', 'rhs')) == [x, h]
    assert values_of(JumpLaw(h * x, {x: -1}), include=('rate',)) == [h * x]
