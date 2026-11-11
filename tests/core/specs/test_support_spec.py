"""Test the `_support_spec` module of `skcomplex.core.specs`."""

from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime, LatticeSpace, StaticTime
from skcomplex.core.specs import SupportSpec


def test_empty_support_accepts_any_class():
    """Test that an empty field supports any class, and `None` in the space field supports no space."""
    assert SupportSpec().get_missing('time', [StaticTime]) == []
    assert SupportSpec(space={None}).get_missing('space', [None, LatticeSpace]) == ['LatticeSpace']


def test_support_accepts_subclasses():
    """Test that a class is supported when it is a subclass of a supported class."""
    assert SupportSpec(time={ContinuousTime}).get_missing('time', [ContinuousTime, DiscreteTime]) == ['DiscreteTime']
