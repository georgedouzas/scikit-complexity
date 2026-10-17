"""Run the examples of every docstring in the package."""

import doctest
import importlib
import pkgutil

import pytest

import skcomplex

MODULES = [module.name for module in pkgutil.walk_packages(skcomplex.__path__, 'skcomplex.')]


@pytest.mark.parametrize('name', MODULES)
def test_docstrings_examples_pass(name):
    """Test that the docstring examples of a module pass."""
    result = doctest.testmod(importlib.import_module(name), optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
