"""Segment of a blood vessel, which supplies oxygen to its site every step, a member transfer of the system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, VisualSpec
from ._vessel_view import VesselView


class VesselElement(Element):
    """Segment of a blood vessel, which supplies oxygen to its site every step, a member transfer of the system.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        supply:
            The oxygen supplied per step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.oncology import VesselElement
        >>> GroupElement('vessels', element=VesselElement, size=2, location=[(0, 0), (0, 1)]).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        supply: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.supply = supply

    @classmethod
    def _get_view(cls) -> type[VesselView]:
        """Return the view of a vessel, `VesselView`."""
        return VesselView
