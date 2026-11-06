"""Cow of the meadow, which the grazing and herding interactions feed and move every week."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, VisualSpec
from ._cow_view import CowView


class CowElement(Element):
    """Cow of the meadow, which the grazing and herding interactions feed and move every week.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        bite:
            The grass a cow eats in a week.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import CowElement
        >>> GroupElement('cows', element=CowElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.4}),
        *,
        bite: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bite = bite

    @classmethod
    def _get_view(cls) -> type[CowView]:
        """Return the view of a cow, `CowView`."""
        return CowView
