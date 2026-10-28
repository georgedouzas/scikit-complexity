"""Test the `types` module of `skcomplex`."""

import enum
import importlib
import inspect
import pkgutil
import typing

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

import skcomplex
from skcomplex.types import Process, Target


def test_targets_are_symbolic_expressions_or_vectors():
    """Test that a law targets a symbolic expression or a vector of them."""
    assert typing.get_args(Target.__value__) == (Expression, FreeModuleElement)


def test_processes_may_be_nothing():
    """Test that a process may be nothing."""
    assert type(None) in typing.get_args(Process.__value__)


def test_no_module_outside_the_types_defines_a_kind_or_a_protocol():
    """Test that no module outside `skcomplex.types` and the models defines an enum or a protocol."""
    defined = []
    for info in pkgutil.walk_packages(skcomplex.__path__, 'skcomplex.'):
        if info.name.startswith('skcomplex.models') or info.name == 'skcomplex.types':
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue
        for value in vars(module).values():
            if (
                inspect.isclass(value)
                and value.__module__ == module.__name__
                and (issubclass(value, enum.Enum) or getattr(value, '_is_protocol', False))
            ):
                defined.append(f'{module.__name__}.{value.__qualname__}')
    assert defined == []
