"""Test the `_geometry_spec` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import GeometrySpec
from skcomplex.core.errors import RewritingError


@pytest.mark.parametrize('spec', [GeometrySpec(symbols={'X': 'jump'}), GeometrySpec(dimension=4)])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a geometry rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec._check()
