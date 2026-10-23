"""Test the traversal of the values a law holds."""

from skcomplex.core.base import symbol, vector
from skcomplex.core.laws import Call, Equation, Event, Jump, Noise, Rate, values_of


def _double(t, value):
    """Return twice a value."""
    return 2 * value


def test_values_of_flattens_calls_vectors_and_mappings():
    """Test that calls give their arguments, vectors their components, and mappings their keys and values."""
    x, y, k = symbol('x'), symbol('y'), symbol('k')
    assert values_of(Rate(x, Call(_double, k * x))) == [x, k * x, None]
    assert values_of(Rate(vector([x, y]), vector([y, -x]))) == [x, y, y, -x, None]
    assert values_of(Event(x, {y: 0, x: k})) == [x, y, 0, x, k, 0, False, None]


def test_values_of_reads_chosen_fields():
    """Test that only the included fields are read, without the excluded ones."""
    x, h = symbol('x'), symbol('h')
    noise = Noise(x, 0.1 * x, process='market')
    assert values_of(noise, include=('value',)) == [0.1 * x]
    assert 'market' not in values_of(noise, exclude=('process',))
    assert values_of(Equation(x == h), include=('lhs', 'rhs')) == [x, h]
    assert values_of(Jump(h * x, {x: -1}), include=('rate',)) == [h * x]
