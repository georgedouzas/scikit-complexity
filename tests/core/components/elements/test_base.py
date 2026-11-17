"""Test the `_base` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import BaseElement, Element, GroupElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class PopulationElement(BaseElement):
    """Population with a state, a parameter, and a visual of its own."""

    visual = VisualSpec('line')

    def __init__(
        self,
        label: str,
        n: Annotated[float, StateSpec()] = 1.0,
        r: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


def test_element_stores_its_states_and_parameters():
    """Test that an element stores its arguments, and its drawing is not one of them."""
    assert PopulationElement('rabbits', n=40).get_params() == {'label': 'rabbits', 'n': 40, 'r': 0.5}


def test_element_prepares_the_visual_of_its_class():
    """Test that preparing an element sets the drawing its class gives as its default visual."""
    element = PopulationElement('rabbits').prepare()
    assert element.default_visual_ == VisualSpec('line')
    assert Element('ball', values={'x': 1.0}).prepare().default_visual_ == VisualSpec()
    assert element.run(SimpleNamespace()).laws_ == []


def test_elements_and_groups_derive_from_the_element_base_only():
    """Test that elements and groups derive from the element base, and interactions and systems do not."""
    for kind in (Element, GroupElement):
        assert issubclass(kind, BaseElement)
    for kind in (BaseInteraction, CompositeSystem):
        assert not issubclass(kind, BaseElement)


def test_get_states_returns_the_states_with_their_specs():
    """Test that an element returns the states it declares with their specs, and no parameter."""
    states = PopulationElement('rabbits').get_states()
    assert list(states) == ['n']
    assert isinstance(states['n'], StateSpec)


class CellElement(Element):
    """Element with one boolean state."""

    def __init__(self, label: str, alive: Annotated[int, StateSpec(FiniteEnumeratedSet([0, 1]))] = 0) -> None:
        super().__init__(label)
        self.alive = alive


class DecayElement(Element):
    """Element that lists the rate law it writes."""

    law_classes = (RateLaw,)

    def __init__(self, label: str, x: Annotated[float, StateSpec()] = 1.0) -> None:
        super().__init__(label)
        self.x = x

    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, -namespace.x)]


class UnlistedElement(Element):
    """Element that writes laws it does not list."""

    def _get_laws(self, namespace):
        return []


def test_element_tags_hold_the_domains_of_its_states_before_it_is_prepared():
    """Test that a boolean group on a lattice has the domain boolean and the space lattice, before it is prepared."""
    group = GroupElement('cells', element=CellElement, space=LatticeSpace((4,)))
    tags = group.get_tags()
    assert (tags.domains, tags.space) == ({'boolean'}, {'lattice'})
    assert not hasattr(group, 'size_')


def test_element_tags_take_law_kinds_from_the_laws_its_class_lists():
    """Test that an element without laws has none, one that lists its laws has their kinds, and others unknown ones."""
    assert CellElement('cell').get_tags().laws == frozenset()
    assert DecayElement('decay').get_tags().laws == {'rate'}
    assert UnlistedElement('free').get_tags().laws is None


def test_variable_group_changes_its_membership():
    """Test that a variable group has the structure of a variable membership."""
    group = GroupElement('cells', element=CellElement, size=0, variable=True, capacity=4)
    assert group.get_tags().structure == {'variable membership'}


class _FallingBall(Element):
    """Ball that wrongly sets its own law of motion."""

    def __init__(self, label: str, x: float = 10.0) -> None:
        super().__init__(label)
        self.x = x

    def _prepare(self, context: LatticeSpace | None = None) -> None:
        """Set a law, which only interactions may do."""
        super()._prepare(context)
        self.laws_ = [RateLaw(var('x'), -9.81)]


def test_prepare_rejects_an_element_that_sets_laws():
    """Test that an element whose _prepare sets laws_ raises an error pointing to interactions."""
    with pytest.raises(ValidationError, match=r"'ball' sets laws_ in _prepare.*interaction"):
        _FallingBall('ball').prepare()


def test_prepare_twice_after_a_run_keeps_working():
    """Test that an element run once, which sets laws_, prepares again without the error."""
    element = Element('ball', values={'x': 1.0}).prepare()
    element.run(SimpleNamespace())
    assert element.prepare() is element
