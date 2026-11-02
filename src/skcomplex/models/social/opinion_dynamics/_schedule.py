"""Triangle of a temporal hypergraph, active in every other half of a period from its own phase."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, VisualSpec


class ScheduleElement(Element):
    """Triangle of a temporal hypergraph, active in every other half of a period from its own phase.

    Args:
        label:
            The label of the triangle or the group.

        visual:
            How the triangle is drawn, or `None` for `default_visual_`.

        phase:
            The phase of the schedule, in radians.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ScheduleElement
        >>> ScheduleElement('triangle', phase=1.0).phase
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phase: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phase = phase
