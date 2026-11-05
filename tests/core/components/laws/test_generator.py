"""Test the `_generator` module of `skcomplex.core.components.laws`."""

import pytest

from skcomplex.core.components.laws import GeneratorLaw
from skcomplex.core.errors import ValidationError


def test_check_names_generators_as_symbolic_only():
    """Test that a function as the operator of a generator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        GeneratorLaw(print).prepare()
