"""Test that the public names of the runtime package are its classes, values, and the functions its clients use."""

import inspect

from skcomplex.engine import runtime as package

FUNCTIONS = {'catalog', 'create', 'describe', 'register_model', 'run', 'summarize'}
VALUES = {'MODELS'}


def test_public_names_are_classes_values_or_functions():
    """Every public name is a class, a value a client reads, or one of the kept functions."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)) or name in FUNCTIONS | VALUES, name


def test_kept_functions_are_public():
    """Every kept function is public and callable."""
    for name in FUNCTIONS:
        assert name in package.__all__
        assert callable(getattr(package, name))


def test_classes_derive_from_a_base_of_the_package():
    """Every observer derives from the observer base and every clock from the clock base."""
    for name in package.__all__:
        item = getattr(package, name)
        if name.endswith('Observer'):
            assert issubclass(item, package.BaseObserver), name
        if name.endswith('Clock'):
            assert issubclass(item, package.BaseClock), name
