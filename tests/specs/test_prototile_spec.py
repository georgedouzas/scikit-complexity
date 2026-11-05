"""Test the `_prototile_spec` module of `skcomplex.specs`."""

import pytest

from skcomplex.errors import RewritingError
from skcomplex.specs import PrototileSpec


@pytest.mark.parametrize('spec', [PrototileSpec('point', [(0, 0)])])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a prototile rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.check()
