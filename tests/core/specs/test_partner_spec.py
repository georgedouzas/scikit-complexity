"""Test the `_partner_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PartnerSpec


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (PartnerSpec(candidates=0.0), 'positive'),
        (PartnerSpec(candidates=True), 'positive'),
        (PartnerSpec(choice='score', score=2), 'score'),
    ],
)
def test_partner_spec_rejects_invalid_settings(spec, message):
    """Test that a distance that is not positive and a bad score raise an error naming them."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_partner_spec_holds_no_value():
    """Test that a selection of partners keeps its candidates and has no clearing value, which only matchings read."""
    assert PartnerSpec(candidates=2.0).candidates == 2.0
    assert not hasattr(PartnerSpec(), 'value')


@pytest.mark.parametrize(
    'arguments',
    [
        {'choice': 'best'},
        {'fallback': 'first'},
        {'candidates': -1.0},
        {'candidates': True},
        {'capacity': 0},
        {'choice': 'score'},
        {'score': print},
    ],
)
def test_partner_spec_check_rejects_unusable_settings(arguments):
    """Test that the check of a partner spec names a choice, fallback, distance, capacity, or score it cannot take."""
    with pytest.raises(ValidationError):
        PartnerSpec(**arguments)._check()
