"""Test the `_geometry_spec` module of `skcomplex.specs`."""

import pytest

from skcomplex.errors import RewritingError
from skcomplex.specs import GeometrySpec


@pytest.mark.parametrize('spec', [GeometrySpec(symbols={'X': 'jump'}), GeometrySpec(dimension=4)])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a geometry rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.prepare()
