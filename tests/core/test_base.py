"""Test the `base` module of `skcomplex.core`."""

import importlib
import inspect
import pkgutil
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import pytest
from sklearn.base import clone

from skcomplex.core.base import BasePart
from skcomplex.core.elements import BaseElement, Element
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.laws import BaseLaw
from skcomplex.core.spaces import BaseNeighborhood, BaseSpace, BaseTime
from skcomplex.core.system import System
from skcomplex.errors import NotCheckedError, ValidationError
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec

SOURCE = str(Path(str(files('skcomplex'))))
PACKAGES = ('skcomplex.core', 'skcomplex.engine', 'skcomplex.experiments', 'skcomplex.analysis')
ARITHMETIC = {
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__truediv__',
    '__rtruediv__',
    '__matmul__',
    '__rmatmul__',
    '__neg__',
    '__pos__',
    '__pow__',
}


class DoublingPart(BasePart):
    """Part that doubles its input after counting its preparations."""

    def __init__(self, label, factor=2):
        super().__init__(label)
        self.factor = factor

    def _check(self):
        """Raise an error for a factor that is not a number."""
        if not isinstance(self.factor, int | float):
            error_msg = f'The factor {self.factor!r} is not a number.'
            raise ValidationError(error_msg)

    def _prepare(self, context=None):
        """Count the preparation and keep the context."""
        self.prepared_ = getattr(self, 'prepared_', 0) + 1
        self.context_ = context

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Set `doubled_`, the input times the factor."""
        self.doubled_ = self.factor * inputs


class SpecifiedPart(BasePart):
    """Part that declares a state, a parameter, and a role, and one plain argument."""

    def __init__(
        self,
        label,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        k: Annotated[float, ParameterSpec()] = 1.0,
        target: Annotated[str, RoleSpec()] = 'a',
        plain=None,
    ):
        super().__init__(label)
        self.x = x
        self.k = k
        self.target = target
        self.plain = plain

    def _check(self):
        """Accept every argument."""

    def _prepare(self, context=None):
        """Prepare nothing."""

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Run nothing."""


class ExtendedPart(SpecifiedPart):
    """Part that takes the state of its base again without annotating it, and adds a parameter first."""

    def __init__(self, label, r: Annotated[float, ParameterSpec()] = 0.5, x=1.0):
        super().__init__(label, x=x)
        self.r = r


class InvalidSpecPart(SpecifiedPart):
    """Part that declares a state of an invalid order."""

    def __init__(self, label, y: Annotated[float, StateSpec(order=0)] = 0.0):
        super().__init__(label)
        self.y = y


def _modules():
    """Return every module of the packages that define the library's classes."""
    for name in PACKAGES:
        package = importlib.import_module(name)
        yield package
        for info in pkgutil.walk_packages(package.__path__, f'{name}.'):
            yield importlib.import_module(info.name)


def _hand_written(cls):
    """Return the magic methods a class defines itself in source code, leaving out those dataclasses generate."""
    names = []
    for name, value in vars(cls).items():
        if not (name.startswith('__') and name.endswith('__')) or not callable(value):
            continue
        code = getattr(value, '__code__', None)
        if code is not None and code.co_filename.startswith(SOURCE):
            names.append(name)
    return names


def test_verbs_check_prepare_and_run():
    """Test that checking marks the part, preparing derives its attributes from a context, and running acts."""
    part = DoublingPart('doubling').check().prepare('context')
    assert (part.prepared_, part.context_) == (1, 'context')
    assert part.run(3).doubled_ == 6


def test_part_must_be_checked_before_it_is_prepared_or_run():
    """Test that preparing or running a part that was not checked raises an error."""
    with pytest.raises(NotCheckedError):
        DoublingPart('doubling').prepare()
    with pytest.raises(NotCheckedError):
        DoublingPart('doubling').run(3)


def test_check_validates_the_label_and_the_arguments():
    """Test that checking rejects a label that is not a string and an argument the part cannot take."""
    with pytest.raises(ValidationError, match='label of DoublingPart must be a string'):
        DoublingPart(1).check()
    with pytest.raises(ValidationError, match='not a number'):
        DoublingPart('doubling', factor='two').check()


def test_check_validates_every_declared_spec():
    """Test that checking a part checks the spec of every annotated argument."""
    with pytest.raises(ValidationError, match='order'):
        InvalidSpecPart('invalid').check()


def test_get_specs_returns_the_annotated_arguments_in_constructor_order():
    """Test that the specs of a part are the annotated arguments of its constructor, in order, without plain ones."""
    specs = SpecifiedPart._get_specs()
    assert list(specs) == ['x', 'k', 'target']
    assert specs['x'] == StateSpec(order=2)
    assert isinstance(specs['target'], RoleSpec)


def test_get_specs_keeps_the_specs_of_the_bases():
    """Test that a subclass keeps the spec of an argument of its base that it takes again without annotating it."""
    specs = ExtendedPart._get_specs()
    assert list(specs) == ['r', 'x']
    assert specs['x'] == StateSpec(order=2)


def test_constructor_only_stores_its_arguments():
    """Test that a part stores its arguments as given, so scikit-learn lists and clones them."""
    part = SpecifiedPart('a', x=[1.0], plain='plain')
    assert part.get_params() == {'k': 1.0, 'label': 'a', 'plain': 'plain', 'target': 'a', 'x': [1.0]}
    cloned = clone(part)
    assert cloned.get_params() == part.get_params()
    assert cloned is not part


def test_element_runs_into_its_laws():
    """Test that running an element sets its laws for the namespace, as interactions and systems do."""

    class StillElement(Element):
        def _get_laws(self, namespace):
            return []

    assert StillElement('a').check().prepare().run(SimpleNamespace()).laws_ == []


def test_parts_derive_from_the_part_base():
    """Test that every part of a system is a part, and a part cannot be made without its hooks."""
    parts = (BaseElement, BaseInteraction, BaseLaw, System, BaseSpace, BaseTime, BaseNeighborhood)
    assert all(issubclass(part, BasePart) for part in parts)
    with pytest.raises(TypeError, match='abstract'):
        BasePart('part')


def test_no_class_defines_magic_methods():
    """Test that no class of the library defines a magic method but `__init__` and quantum arithmetic."""
    found = {}
    for module in _modules():
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            extra = set(_hand_written(cls)) - {'__init__'}
            if cls.__qualname__ == 'Operator' and module.__name__.startswith(
                'skcomplex.engine.problems._operator',
            ):
                extra -= ARITHMETIC
            if extra:
                found[f'{module.__name__}.{cls.__qualname__}'] = sorted(extra)
    assert found == {}, found
