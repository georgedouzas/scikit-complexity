"""Test the `affine_spec` module of `skcomplex.models.mathematics.fractals`."""

import numpy as np
import pytest

from skcomplex.models.mathematics.fractals import AffineSpec, RewritingError


@pytest.mark.parametrize('spec', [AffineSpec(translation=(1, 0, 0)), AffineSpec(matrix=np.eye(2))])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that an affine map rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.check()
