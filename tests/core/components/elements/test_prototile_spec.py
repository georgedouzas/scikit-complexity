"""Test the `_prototile_spec` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import PrototileSpec
from skcomplex.core.errors import RewritingError


@pytest.mark.parametrize('spec', [PrototileSpec('point', [(0, 0)])])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a prototile rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec._check()
