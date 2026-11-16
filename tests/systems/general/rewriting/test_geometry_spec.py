"""Test the `_geometry_spec` module of `skcomplex.systems.general.rewriting`."""

import pytest

from skcomplex.core.errors import RewritingError
from skcomplex.systems.general.rewriting import GeometrySpec


@pytest.mark.parametrize('spec', [GeometrySpec(symbols={'X': 'jump'}), GeometrySpec(dimension=4)])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that a geometry rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec._check()
