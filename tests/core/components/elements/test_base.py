"""Test the `_base` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import BaseElement, Element
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import VisualSpec
from skcomplex.core.systems import System


class PopulationElement(Element):
    """Population with two values and a visual of its own."""

    visual = VisualSpec('line')

    def __init__(self, label: str, n: float = 1.0, r: float = 0.5) -> None:
        super().__init__(label)
        self.n = n
        self.r = r


def test_element_stores_its_states_and_parameters():
    """Test that an element stores its arguments, and its drawing is not one of them."""
    assert PopulationElement('rabbits', n=40).get_params() == {'label': 'rabbits', 'n': 40, 'r': 0.5}


def test_element_prepares_the_visual_of_its_class():
    """Test that preparing an element sets the drawing its class gives as its default visual."""
    element = PopulationElement('rabbits').prepare()
    assert element.default_visual_ == VisualSpec('line')
    assert Element('ball', values={'x': 1.0}).prepare().default_visual_ == VisualSpec()


def test_elements_derive_from_the_element_base_only():
    """Test that elements derive from the element base, and interactions and systems do not."""
    assert issubclass(Element, BaseElement)
    for kind in (BaseInteraction, System):
        assert not issubclass(kind, BaseElement)


class CellElement(Element):
    """Cells with one boolean value."""

    def __init__(self, label: str, alive: Annotated[int, {0, 1}] = 0, space: LatticeSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.alive = alive


def test_element_tags_hold_the_domains_of_its_states_before_it_is_prepared():
    """Test that boolean cells on a lattice have the domain boolean and the space lattice, before they are prepared."""
    cells = CellElement('cells', space=LatticeSpace((4,)))
    tags = cells.get_tags()
    assert (tags.domains, tags.space) == ({'boolean'}, {'lattice'})
    assert not hasattr(cells, 'size_')


def test_element_tags_hold_no_law_kinds():
    """Test that an element, which sets no laws, has no law kinds."""
    assert CellElement('cell').get_tags().laws == frozenset()


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
