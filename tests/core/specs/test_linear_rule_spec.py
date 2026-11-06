"""Test the linear rule spec."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LinearRuleSpec


def test_characteristic_reads_the_field():
    """Test that the characteristic is the prime of a finite field and 0 for the rationals."""
    assert LinearRuleSpec((1, 1), 'GF(3)').characteristic == 3
    assert LinearRuleSpec((1, 1), 'GF( 5 )').characteristic == 5
    assert LinearRuleSpec((1, 1), 'QQ').characteristic == 0


def test_check_rejects_an_unknown_field():
    """Test that a field that is neither the rationals nor a prime field is rejected."""
    with pytest.raises(ValidationError, match='Unknown field'):
        LinearRuleSpec((1,), 'GF(x)').prepare()
    with pytest.raises(ValidationError, match='Unknown field'):
        LinearRuleSpec((1,), 'RR').prepare()
