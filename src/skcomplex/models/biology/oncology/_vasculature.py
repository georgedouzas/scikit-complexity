"""Vasculature of a tumour, which sets its carrying capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class VasculatureElement(Element):
    """Vasculature of a tumour, which sets its carrying capacity.

    Args:
        label:
            The label of the vasculature.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        K:
            The initial carrying capacity.

    Examples:
        >>> from skcomplex.models.biology.oncology import VasculatureElement
        >>> VasculatureElement('vessels', K=12.7).K
        12.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        K: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.K = K
