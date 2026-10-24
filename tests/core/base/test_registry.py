"""Test the `_registry` module of `skcomplex.core.base`."""

import pytest

from skcomplex.core.base import Registry


def test_registry_resolve_unknown_key_lists_keys():
    """Test that resolving an unknown key names the registered keys."""
    registry = Registry[int]('number')
    registry.register('one', 1)
    with pytest.raises(KeyError, match='Registered keys are: one'):
        registry.resolve('two')
