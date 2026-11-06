"""Test the `_kirschner_panetta_tumour` module of `skcomplex.systems.biology.oncology`."""

from skcomplex.systems.biology.oncology import (
    KirschnerPanettaTumourElement,
)


def test_kirschner_panetta_is_not_registered_twice():
    """Test that the scaled tumour of the immunotherapy system is a new element, not the book's dimensional one."""
    assert KirschnerPanettaTumourElement('site').alpha1 == 2e7
