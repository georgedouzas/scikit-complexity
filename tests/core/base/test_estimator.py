"""Test the verbs of the framework's estimator and the rule that behaviour is named, not magic."""

import importlib
import inspect
import pkgutil
from importlib.resources import files
from pathlib import Path

from skcomplex.core.base import BaseEstimator

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
