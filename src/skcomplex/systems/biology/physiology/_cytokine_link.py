"""Link between two cells of a layer, a fixed connectivity `a` and an adaptive cytokine coupling `kappa`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CytokineLinkElement(Element):
    """Link between two cells of a layer, a fixed connectivity `a` and an adaptive cytokine coupling `kappa`.

    Args:
        label:
            The label of the link or of the group on the edges.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        kappa:
            The initial adaptive coupling, between -1 and 1.

        a:
            The fixed connectivity, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CytokineLinkElement
        >>> CytokineLinkElement('link', a=1.0).a
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec(domain=RealSet.closed(-1, 1))] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa
        self.a = a
