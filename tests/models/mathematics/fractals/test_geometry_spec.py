"""Test the `geometry_spec` module of `skcomplex.models.mathematics.fractals`."""

import pytest

from skcomplex.models.mathematics.fractals import GeometrySpec, RewritingError


@pytest.mark.parametrize('spec', [GeometrySpec(symbols={'X': 'jump'}), GeometrySpec(dimension=4)])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a geometry rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.check()
