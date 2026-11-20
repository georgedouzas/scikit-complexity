"""Test the `_parameter_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.components._parameter_spec import _ParameterSpec
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (_ParameterSpec(per='inputs', vector=True), 'per entry'),
        (_ParameterSpec(per='inputs', adaptive=True), 'per entry'),
        (_ParameterSpec(bounds=(0, 1)), 'adaptive parameter'),
        (_ParameterSpec(adaptive=True, bounds=(1, 0)), 'lower bound at most the upper one'),
        (_ParameterSpec(shape=()), 'nonempty tuple of role names'),
        (_ParameterSpec(shape=('target',), vector=True), 'neither a vector'),
        (_ParameterSpec(shape=('target',), per='inputs'), 'per entry'),
    ],
)
def test_parameter_spec_rejects_invalid_declarations(spec, message):
    """Test that a parameter per role that is a vector or adaptive, invalid bounds, and bad shapes are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_parameter_spec_accepts_bounds_of_an_adaptive_parameter():
    """Test that an adaptive parameter takes bounds."""
    spec = _ParameterSpec(adaptive=True, bounds=(0, 1))
    spec._check()


def test_parameter_spec_accepts_a_shape_of_role_names():
    """Test that a parameter takes the roles of its shape, a role repeated for a tensor."""
    assert _ParameterSpec(shape=('target',) * 3, adaptive=True).shape == ('target', 'target', 'target')
