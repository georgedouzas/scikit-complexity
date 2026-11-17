"""Test the `base` module of `skcomplex.core.components`."""

from types import SimpleNamespace
from typing import Annotated

import pytest
from sklearn.base import clone

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import BaseElement, Element
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import BaseNeighborhood, BaseSpace, BaseTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class DoublingComponent(BaseComponent):
    """Component that doubles its input after counting its preparations."""

    def __init__(self, label, factor=2):
        super().__init__(label)
        self.factor = factor

    def _prepare(self, context=None):
        """Check the factor, then count the preparation and keep the context."""
        super()._prepare(context)
        if not isinstance(self.factor, int | float):
            error_msg = f'The factor {self.factor!r} is not a number.'
            raise ValidationError(error_msg)
        self.prepared_ = getattr(self, 'prepared_', 0) + 1
        self.context_ = context

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Set `doubled_`, the input times the factor."""
        self.doubled_ = self.factor * inputs


class SpecifiedComponent(BaseComponent):
    """Component that declares a state, a parameter, and a role, and one plain argument."""

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

    def _prepare(self, context=None):
        """Check the label and the declared specs, and prepare nothing else."""
        super()._prepare(context)

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Run nothing."""


class ExtendedComponent(SpecifiedComponent):
    """Component that takes the state of its base again without annotating it, and adds a parameter first."""

    def __init__(self, label, r: Annotated[float, ParameterSpec()] = 0.5, x=1.0):
        super().__init__(label, x=x)
        self.r = r


class InvalidSpecComponent(SpecifiedComponent):
    """Component that declares a state of an invalid order."""

    def __init__(self, label, y: Annotated[float, StateSpec(order=0)] = 0.0):
        super().__init__(label)
        self.y = y


def test_prepare_and_run():
    """Test that preparing derives the attributes of the component from a context, and running acts."""
    component = DoublingComponent('doubling').prepare('context')
    assert (component.prepared_, component.context_) == (1, 'context')
    assert component.run(3).doubled_ == 6


def test_prepare_checks_the_label_and_the_arguments():
    """Test that preparing rejects a label that is not a string and an argument the component cannot take."""
    with pytest.raises(ValidationError, match='label of DoublingComponent must be a string'):
        DoublingComponent(1).prepare()
    with pytest.raises(ValidationError, match='not a number'):
        DoublingComponent('doubling', factor='two').prepare()


def test_prepare_checks_every_declared_spec():
    """Test that preparing a component checks the spec of every annotated argument."""
    with pytest.raises(ValidationError, match='order'):
        InvalidSpecComponent('invalid').prepare()


def test_get_specs_returns_the_annotated_arguments_in_constructor_order():
    """Test that the specs of a component are the annotated arguments of its constructor, in order, and no others."""
    specs = SpecifiedComponent._get_specs()
    assert list(specs) == ['x', 'k', 'target']
    assert specs['x'] == StateSpec(order=2)
    assert isinstance(specs['target'], RoleSpec)


def test_get_specs_keeps_the_specs_of_the_bases():
    """Test that a subclass keeps the spec of an argument of its base that it takes again without annotating it."""
    specs = ExtendedComponent._get_specs()
    assert list(specs) == ['r', 'x']
    assert specs['x'] == StateSpec(order=2)


def test_constructor_only_stores_its_arguments():
    """Test that a component stores its arguments as given, so scikit-learn lists and clones them."""
    component = SpecifiedComponent('a', x=[1.0], plain='plain')
    assert component.get_params() == {'k': 1.0, 'label': 'a', 'plain': 'plain', 'target': 'a', 'x': [1.0]}
    cloned = clone(component)
    assert cloned.get_params() == component.get_params()
    assert cloned is not component


def test_element_runs_into_its_laws():
    """Test that running an element sets its laws for the namespace, as interactions and systems do."""

    class StillElement(Element):
        def _get_laws(self, namespace):
            return []

    assert StillElement('a').prepare().run(SimpleNamespace()).laws_ == []


def test_components_derive_from_the_component_base():
    """Test that every component of a system is a component, and a component cannot be made without its hooks."""
    components = (BaseElement, BaseInteraction, BaseLaw, CompositeSystem, BaseSpace, BaseTime, BaseNeighborhood)
    assert all(issubclass(component, BaseComponent) for component in components)
    with pytest.raises(TypeError, match='abstract'):
        BaseComponent('component')
