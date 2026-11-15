"""Test the `_affine_spec` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest

from skcomplex.core.components.laws import AffineSpec
from skcomplex.core.errors import RewritingError


@pytest.mark.parametrize('spec', [AffineSpec(translation=(1, 0, 0)), AffineSpec(matrix=np.eye(2))])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that an affine map rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec._check()
