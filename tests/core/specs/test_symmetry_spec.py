"""Test the `_symmetry_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import SymmetrySpec


@pytest.mark.parametrize(
    ('spec', 'message'),
    [(SymmetrySpec('', lambda s: {}), 'empty label'), (SymmetrySpec('rotation', 'turn'), 'generator')],
)
def test_check_symmetry_spec_rejects_invalid_settings(spec, message):
    """Test that a symmetry with an empty label or a generator that is not callable raises an error naming it."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
