"""Test the `_parameter_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (ParameterSpec(per='inputs', vector=True), 'per entry'),
        (ParameterSpec(per='inputs', adaptive=True), 'per entry'),
        (ParameterSpec(bounds=(0, 1)), 'adaptive parameter'),
        (ParameterSpec(adaptive=True, bounds=(1, 0)), 'lower bound at most the upper one'),
        (ParameterSpec(shape=()), 'nonempty tuple of role names'),
        (ParameterSpec(shape=('target',), vector=True), 'neither a vector'),
        (ParameterSpec(shape=('target',), per='inputs'), 'per entry'),
    ],
)
def test_parameter_spec_rejects_invalid_declarations(spec, message):
    """Test that a parameter per role that is a vector or adaptive, invalid bounds, and bad shapes are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_parameter_spec_accepts_bounds_of_an_adaptive_parameter():
    """Test that an adaptive parameter takes bounds."""
    spec = ParameterSpec(adaptive=True, bounds=(0, 1))
    spec._check()


def test_parameter_spec_accepts_a_shape_of_role_names():
    """Test that a parameter takes the roles of its shape, a role repeated for a tensor."""
    assert ParameterSpec(shape=('target',) * 3, adaptive=True).shape == ('target', 'target', 'target')
