"""Test the `_registry` module of `skcomplex.core.base`."""

import pytest

from skcomplex.core.base import Registry


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
