"""Test the `_power_series_spec` module of `skcomplex.core.specs`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import GF, QQ, RR

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import EntrySpec, PowerSeriesSpec


def test_product_is_the_truncated_cauchy_product():
    """Test that the product of two series keeps the coefficients below the order of the exact product."""
    ring = PowerSeriesSpec(5)
    rng = np.random.default_rng(0)
    first, second = rng.integers(-3, 4, (2, 7, 5)).astype(float)
    product = ring.multiply(first, second)
    for row in range(7):
        exact = ring.series_of(first[row]) * ring.series_of(second[row])
        np.testing.assert_array_equal(product[row], ring.coefficients_of(exact))


def test_prime_field_reduces_coefficients():
    """Test that a ring over a prime field reduces sums and products modulo its characteristic."""
    ring = PowerSeriesSpec(3, 'GF(3)')
    assert ring.characteristic == 3
    assert ring.reduce([4, -1, 3]).tolist() == [1.0, 2.0, 0.0]
    assert ring.multiply([2, 1, 0], [2, 2, 0]).tolist() == [1.0, 0.0, 2.0]
    assert ring.sage_field is GF(3)
    assert list(ring.alphabet) == [0, 1, 2]


def test_coefficients_round_trip_through_sage():
    """Test that coefficients become a Sage series of the order and come back unchanged."""
    ring = PowerSeriesSpec(4, variable='s')
    series = ring.series_of([1, 0, -2, 5])
    assert series.prec() == 4
    assert str(series) == '1 - 2*s^2 + 5*s^3 + O(s^4)'
    assert ring.coefficients_of(series).tolist() == [1.0, 0.0, -2.0, 5.0]
    assert ring.coefficients_of(1 + ring.ring.gen()).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_fields_and_alphabets():
    """Test that the rationals and the reals are the alphabets of their own coefficients."""
    assert PowerSeriesSpec(field='QQ').alphabet is QQ
    assert PowerSeriesSpec(field='RR').alphabet is RR
    assert PowerSeriesSpec().characteristic == 0


def test_entries_of_power_series_check_coefficients():
    """Test that an entry of a power series state checks its coefficients against the alphabet of the ring."""
    entry = EntrySpec('cells.value', 0, (4, 3), PowerSeriesSpec(3, 'GF(2)'), members=4)
    assert entry.contains(np.eye(4, 3))
    assert not entry.contains(np.full((4, 3), 2.0))


@pytest.mark.parametrize(('order', 'field'), [(0, 'QQ'), (2.5, 'QQ'), (True, 'QQ'), (3, 'ZZ'), (3, 'GF(x)')])
def test_invalid_rings_raise(order, field):
    """Test that an order that is not a positive integer or an unknown field raises a validation error."""
    with pytest.raises(ValidationError):
        PowerSeriesSpec(order, field)._check()
