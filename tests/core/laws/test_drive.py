"""Test the `_drive` module of `skcomplex.core.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import DriveLaw
from skcomplex.errors import ValidationError


def test_drive_targets_its_lattice_state():
    """Test that a drive targets the one lattice state it changes."""
    height = var('height')
    assert DriveLaw(height, lambda values, *, random: values + 1).check().prepare().targets_ == (height,)


def test_drive_needs_a_callable_function():
    """Test that a drive rejects a function that cannot be called."""
    with pytest.raises(ValidationError, match='callable'):
        DriveLaw(var('height'), 1.0).check()
