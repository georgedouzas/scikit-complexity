"""Resident of the Schelling system, who moves to an empty cell while too few neighbours share its kind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._resident_view import ResidentView


class ResidentElement(Element):
    """Resident of the Schelling system, who moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        kind:
            The kind of the resident, 0 or 1.

        threshold:
            The smallest share of neighbours of the same kind that keeps a resident in place.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.segregation import ResidentElement
        >>> group = GroupElement('residents', element=ResidentElement, size=10, values={'kind': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('residents', 'ResidentElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'kind', 'color_range': (0.0, 1.0)}),
        *,
        kind: Annotated[float, StateSpec(domain=NN)] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ResidentView
