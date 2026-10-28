"""Test that the public names of the backends package are its classes and the functions its clients use."""

import inspect

from skcomplex.engine import backends as package

FUNCTIONS = {'reads_reductions', 'resolve_backend'}


def test_public_names_are_classes_or_functions():
    """Every public name is a class or one of the kept functions."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)) or name in FUNCTIONS, name


def test_kept_functions_are_public():
    """Every kept function is public and callable."""
    for name in FUNCTIONS:
        assert name in package.__all__
        assert callable(getattr(package, name))
