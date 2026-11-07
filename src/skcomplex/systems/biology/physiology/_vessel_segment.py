"""Vessel segment on an edge of a transport network, whose conductance adapts to the flow it carries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class VesselSegmentElement(Element):
    """Vessel segment on an edge of a transport network, whose conductance adapts to the flow it carries.

    Args:
        label:
            The label of the segment or of the group on the edges.

        visual:
            How the segment is drawn, or `None` for `default_visual_`.

        C:
            The initial conductance.

    Examples:
        >>> from skcomplex.systems.biology.physiology import VesselSegmentElement
        >>> VesselSegmentElement('vessel', C=2.0).C
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.C = C
