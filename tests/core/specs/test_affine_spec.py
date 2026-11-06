"""Test the `_affine_spec` module of `skcomplex.core.specs`."""

import numpy as np
import pytest

from skcomplex.core.errors import RewritingError
from skcomplex.core.specs import AffineSpec


@pytest.mark.parametrize('spec', [AffineSpec(translation=(1, 0, 0)), AffineSpec(matrix=np.eye(2))])
def test_invalid_fields_raise_rewriting_errors(spec):
    """Test that an affine map rejects invalid fields with a rewriting error."""
    with pytest.raises(RewritingError):
        spec.prepare()
