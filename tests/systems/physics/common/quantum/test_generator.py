"""Test the `_generator` module of `skcomplex.systems.physics.common.quantum`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.common.quantum import GeneratorLaw


def test_check_names_generators_as_symbolic_only():
    """Test that a function as the operator of a generator raises an error naming it as symbolic only."""
    with pytest.raises(ValidationError, match=r'need symbolic operators.*`RateLaw`'):
        GeneratorLaw(print).prepare()
