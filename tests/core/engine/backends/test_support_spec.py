"""Test the `_support_spec` module of `skcomplex.core.engine.backends`."""

from skcomplex.core.components.spaces import LatticeSpace, NetworkSpace, PhaseSpace
from skcomplex.core.engine.backends import SupportSpec


def test_empty_support_accepts_any_class():
    """Test that an empty field supports any class, and `None` in the space field supports no space."""
    assert SupportSpec().get_missing('space', [LatticeSpace]) == []
    assert SupportSpec(space={None}).get_missing('space', [None, LatticeSpace]) == ['LatticeSpace']


def test_support_accepts_subclasses():
    """Test that a class is supported when it is a subclass of a supported class."""
    assert SupportSpec(space={LatticeSpace}).get_missing('space', [LatticeSpace, NetworkSpace]) == ['NetworkSpace']
    assert SupportSpec(space={object}).get_missing('space', [PhaseSpace]) == []
