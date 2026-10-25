"""Test the `_utils` module of `skcomplex.core.base`."""

import numpy as np
import pytest

from skcomplex.core.base import (
    Registry,
    wrap_angle,
)


def test_registry_resolve_unknown_key_lists_keys():
    """Test that resolving an unknown key names the registered keys."""
    registry = Registry[int]('number')
    registry.register('one', 1)
    with pytest.raises(KeyError, match='Registered keys are: one'):
        registry.resolve('two')


def test_registry_loader_registers_an_unknown_key():
    """Test that a registered loader is asked for an unknown key, and that a key it does not register still fails."""
    registry = Registry[int]('number')
    registry.register_loader(lambda key: registry.register(key, 2) if key == 'two' else None)
    assert registry.resolve('two') == 2
    with pytest.raises(KeyError, match="'three'"):
        registry.resolve('three')


def test_wrap_angle_maps_into_half_open_turn():
    """Test that wrap_angle maps any real value into (-π, π]."""
    values = wrap_angle(np.array([0.1, -0.1, 7.0, -7.0, np.pi, -np.pi]))
    np.testing.assert_allclose(values, [0.1, -0.1, 7.0 - 2 * np.pi, -7.0 + 2 * np.pi, np.pi, np.pi])


def test_wrap_angle_keeps_angles_inside_exactly():
    """Test that wrap_angle returns angles already in (-π, π] bit for bit, so boundary laws see no change."""
    values = np.linspace(-3.14, np.pi, 1001)
    assert np.array_equal(wrap_angle(values), values)
