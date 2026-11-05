"""Test that the public names of the backends package are classes under its base or values."""

import inspect

from skcomplex.core.engine import backends as package
from skcomplex.core.engine.backends import BaseBackend
from skcomplex.core.specs import BaseSpec


def test_public_names_are_classes_under_the_bases():
    """Every public name is a backend or a value a backend returns."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BaseBackend | BaseSpec), name


def test_backends_name_themselves_and_their_support():
    """Every concrete backend answers the hooks of its name and of what it supports."""
    for backend in (package.NumpyBackend, package.NumbaBackend, package.JaxBackend):
        assert isinstance(backend._get_name(), str)
        assert backend._get_support().laws
