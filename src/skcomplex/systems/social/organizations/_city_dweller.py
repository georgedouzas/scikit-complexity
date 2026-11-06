"""Resident of a growing city, with an angle of similarity, an age, and the meetings of the last step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec

MEETINGS_VISUAL = VisualSpec('sites', {'states': ['meetings']})


class CityDwellerElement(Element):
    """Resident of a growing city, with an angle of similarity, an age, and the meetings of the last step.

    Args:
        label:
            The label of the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        angle:
            The angle of the resident on the circle, its similarity coordinate.

        age:
            The number of steps since the resident joined.

        degree:
            The number of neighbours of the resident at the start of the last step.

        meetings:
            The number of neighbours the resident met in the last step.

    Examples:
        >>> from skcomplex.systems.social.organizations import CityDwellerElement
        >>> CityDwellerElement('residents', angle=1.5).angle
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = MEETINGS_VISUAL,
        *,
        angle: Annotated[float, StateSpec()] = 0.0,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        meetings: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.angle = angle
        self.age = age
        self.degree = degree
        self.meetings = meetings
