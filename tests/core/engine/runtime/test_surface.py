"""Test that the public names of the runtime package are classes, and that it resolves no system name."""

import inspect

import pytest

from skcomplex.core.engine import runtime as package
from skcomplex.core.errors import ValidationError

CATALOG_NAMES = {
    'SYSTEMS',
    'SystemEntry',
    'catalog',
    'create_simulation',
    'create_system',
    'describe',
    'register_system',
    'summarize',
}


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_no_catalog_in_the_engine():
    """No public name registers, lists, describes, creates, or runs systems by name."""
    assert not CATALOG_NAMES & set(package.__all__)
    assert not CATALOG_NAMES & set(dir(package))


def test_simulation_rejects_a_system_name():
    """A simulation of a system name raises an error that says to create the system with the catalog."""
    with pytest.raises(ValidationError, match=r'takes a system or a compiled problem, not .sir.\. Create a system'):
        package.Simulation('sir').prepare()


def test_classes_derive_from_a_base_of_the_package():
    """Every observer derives from the observer base and every clock from the clock base."""
    for name in package.__all__:
        item = getattr(package, name)
        if name.endswith('Observer'):
            assert issubclass(item, package.BaseObserver), name
        if name.endswith('Clock'):
            assert issubclass(item, package.BaseClock), name
