"""Test the `_utils` module of `skcomplex.core.base`."""

import importlib
import inspect
import pkgutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from skcomplex.core.base import (
    BaseEstimator,
    Registry,
    wrap_angle,
)

SOURCE = str(Path(str(files('skcomplex'))))
PACKAGES = ('skcomplex.core', 'skcomplex.engine', 'skcomplex.formalisms')
ARITHMETIC = {
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__truediv__',
    '__rtruediv__',
    '__matmul__',
    '__rmatmul__',
    '__neg__',
    '__pos__',
    '__pow__',
}


class _DoublingEstimator(BaseEstimator):
    """Estimator that doubles its input after counting its preparations and releases."""

    def __init__(self, factor=2):
        self.factor = factor

    def _prepare(self):
        """Count the preparation."""
        self.prepared_ = getattr(self, 'prepared_', 0) + 1

    def _run(self, value):
        """Return the input times the factor."""
        return self.factor * value

    def _release(self):
        """Mark the release."""
        self.released_ = True


def _modules():
    """Return every module of the packages that define the library's classes."""
    for name in PACKAGES:
        package = importlib.import_module(name)
        yield package
        for info in pkgutil.walk_packages(package.__path__, f'{name}.'):
            yield importlib.import_module(info.name)


def _hand_written(cls):
    """Return the magic methods a class defines itself in source code, leaving out those dataclasses generate."""
    names = []
    for name, value in vars(cls).items():
        if not (name.startswith('__') and name.endswith('__')) or not callable(value):
            continue
        code = getattr(value, '__code__', None)
        if code is not None and code.co_filename.startswith(SOURCE):
            names.append(name)
    return names


def test_verbs_prepare_run_and_release():
    """Test that preparing checks and derives, running acts, and the final run releases."""
    estimator = _DoublingEstimator().prepare()
    assert estimator.prepared_ == 1
    assert estimator.run(3) == 6
    assert not hasattr(estimator, 'released_')
    assert estimator.run(4, final=True) == 8
    assert estimator.released_


def test_no_class_defines_magic_methods():
    """Test that no class of the library defines a magic method but `__init__` and quantum arithmetic."""
    found = {}
    for module in _modules():
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            extra = set(_hand_written(cls)) - {'__init__'}
            if cls.__qualname__ == 'Operator' and module.__name__.startswith('skcomplex.formalisms.quantum'):
                extra -= ARITHMETIC
            if extra:
                found[f'{module.__name__}.{cls.__qualname__}'] = sorted(extra)
    assert found == {}, found


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


def test_registry_keys_keep_registration_order():
    """Test that the keys of a registry list every registered key in registration order."""
    registry = Registry[int]('number')
    registry.register('two', 2)
    registry.register('one', 1)
    assert registry.keys == ['two', 'one']


def test_wrap_angle_maps_into_half_open_turn():
    """Test that wrap_angle maps any real value into (-π, π]."""
    values = wrap_angle(np.array([0.1, -0.1, 7.0, -7.0, np.pi, -np.pi]))
    np.testing.assert_allclose(values, [0.1, -0.1, 7.0 - 2 * np.pi, -7.0 + 2 * np.pi, np.pi, np.pi])


def test_wrap_angle_keeps_angles_inside_exactly():
    """Test that wrap_angle returns angles already in (-π, π] bit for bit, so boundary laws see no change."""
    values = np.linspace(-3.14, np.pi, 1001)
    assert np.array_equal(wrap_angle(values), values)
