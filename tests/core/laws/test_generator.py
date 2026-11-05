"""Test the `_generator` module of `skcomplex.core.laws`."""

import pytest

from skcomplex.core.laws import GeneratorLaw
from skcomplex.errors import ValidationError


def test_check_names_generators_as_symbolic_only():
    """Test that a function as the operator of a generator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        GeneratorLaw(print).prepare()
