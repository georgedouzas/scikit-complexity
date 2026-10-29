"""Test the `_base` module of `skcomplex.core.elements`."""

from types import SimpleNamespace
from typing import Annotated

from skcomplex.core.elements import BaseElement, Element, GroupElement
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.system import System
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class PopulationElement(BaseElement):
    """Population with a state, a parameter, and a visual of its own."""

    def __init__(
        self,
        label: str,
        n: Annotated[float, StateSpec()] = 1.0,
        r: Annotated[float, ParameterSpec()] = 0.5,
        visual: VisualSpec | None = VisualSpec('line'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.r = r

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


def test_element_stores_its_states_parameters_and_visual():
    """Test that an element stores its arguments, which scikit-learn lists with the fields of its visual spec."""
    assert PopulationElement('rabbits', n=40).get_params() == {
        'label': 'rabbits',
        'n': 40,
        'r': 0.5,
        'visual': VisualSpec('line'),
        'visual__kind': 'line',
        'visual__options': {},
    }


def test_element_prepares_the_plain_visual():
    """Test that preparing an element sets the plain visual as its default visual, whatever its own visual."""
    element = PopulationElement('rabbits').check().prepare()
    assert element.default_visual_ == VisualSpec()
    assert element.run(SimpleNamespace()).laws_ == []


def test_elements_and_groups_derive_from_the_element_base_only():
    """Test that elements and groups derive from the element base, and interactions and systems do not."""
    for kind in (Element, GroupElement):
        assert issubclass(kind, BaseElement)
    for kind in (BaseInteraction, System):
        assert not issubclass(kind, BaseElement)
