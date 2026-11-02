"""Test the `_dissipator` module of `skcomplex.core.laws`."""

import pytest

from skcomplex.core.laws import DissipatorLaw
from skcomplex.errors import ValidationError


def test_check_names_dissipators_as_symbolic_only():
    """Test that a function as the operator of a dissipator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        DissipatorLaw(print).check()
