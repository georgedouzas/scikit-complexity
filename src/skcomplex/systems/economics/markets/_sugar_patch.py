"""Cell of the landscape whose sugar grows back by a fixed amount a step up to its capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._sugar_patch_view import SugarPatchView


class SugarPatchElement(Element):
    """Cell of the landscape whose sugar grows back by a fixed amount a step up to its capacity.

    The growback is a `RegrowthInteraction` of the group of patches.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The sugar on the patch.

        capacity:
            The most sugar the patch holds.

        growback:
            The sugar that grows back in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import SugarPatchElement
        >>> GroupElement('patches', element=SugarPatchElement, size=4, values={'capacity': 2.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar', 'color_range': (0.0, 4.0)}),
        *,
        sugar: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        growback: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.capacity = capacity
        self.growback = growback

    @classmethod
    def _get_view(cls) -> type[SugarPatchView]:
        """Return the view of a patch, with its sugar, capacity, and growback."""
        return SugarPatchView
