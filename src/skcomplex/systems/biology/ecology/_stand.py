"""Stand of trees of one successional group, measured by its stem volume."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class StandElement(Element):
    """Stand of trees of one successional group, measured by its stem volume.

    Args:
        label:
            The label of the stand.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        volume:
            The initial stem volume.

        mature:
            The stem volume of the group in a mature forest.

    Examples:
        >>> from skcomplex.systems.biology.ecology import StandElement
        >>> StandElement('keora', mature=196.11).mature
        196.11
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        volume: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.volume = volume
        self.mature = mature
