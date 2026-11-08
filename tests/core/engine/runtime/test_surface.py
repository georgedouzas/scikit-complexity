"""Test that the public names of the runtime package are classes or `simulate`, and that it resolves no system name."""

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


def test_public_names_are_classes_or_simulate():
    """Every public name is a class, apart from `simulate`, the function that runs a system."""
    for name in set(package.__all__) - {'simulate'}:
        assert inspect.isclass(getattr(package, name)), name
    assert inspect.isfunction(package.simulate)


def test_no_catalog_in_the_engine():
    """No public name registers, lists, describes, creates, or runs systems by name."""
    assert not CATALOG_NAMES & set(package.__all__)
    assert not CATALOG_NAMES & set(dir(package))


def test_simulation_is_private():
    """The simulation that `simulate` runs is not public."""
    assert 'Simulation' not in package.__all__
    assert not hasattr(package, 'Simulation')


def test_simulate_rejects_a_system_name():
    """Simulating a system name raises an error that names what it took."""
    with pytest.raises(ValidationError, match=r"simulate takes a system, not 'sir'"):
        package.simulate('sir', until=1)


def test_classes_derive_from_a_base_of_the_package():
    """Every observer derives from the observer base and every clock from the clock base."""
    for name in package.__all__:
        item = getattr(package, name)
        if name.endswith('Observer'):
            assert issubclass(item, package.BaseObserver), name
        if name.endswith('Clock'):
            assert issubclass(item, package.BaseClock), name
