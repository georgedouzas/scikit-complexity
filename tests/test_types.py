"""Test the `types` module of `skcomplex`."""

import enum
import importlib
import inspect
import pkgutil

import skcomplex


def test_no_module_outside_the_types_defines_a_kind_or_a_protocol():
    """Test that no module but a `types` module and the models defines an enum or a protocol."""
    defined = []
    for info in pkgutil.walk_packages(skcomplex.__path__, 'skcomplex.'):
        if info.name.startswith('skcomplex.models') or info.name.endswith('.types'):
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
