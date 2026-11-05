"""Test that the public names of the viz package are its classes, each derived from a base of the package."""

import inspect

from skcomplex.core.engine import viz as package
from skcomplex.core.engine.runtime import BaseObserver


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_classes_derive_from_a_base():
    """Every visual derives from the visual base, and the Rerun observer from the observer base."""
    for name in package.__all__:
        base = BaseObserver if name == 'RerunObserver' else package.BaseVisual
        assert issubclass(getattr(package, name), base), name


def test_verbs_take_one_input():
    """A visual prepares on a problem and runs on a frame."""
    assert list(inspect.signature(package.BaseVisual.prepare).parameters) == ['self', 'problem']
    assert list(inspect.signature(package.BaseVisual.run).parameters) == ['self', 'frame', 'final']
