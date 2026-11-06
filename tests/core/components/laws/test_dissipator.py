"""Test the `_dissipator` module of `skcomplex.core.components.laws`."""

import pytest

from skcomplex.core.components.laws import DissipatorLaw
from skcomplex.core.errors import ValidationError


def test_check_names_dissipators_as_symbolic_only():
    """Test that a function as the operator of a dissipator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        DissipatorLaw(print).prepare()
