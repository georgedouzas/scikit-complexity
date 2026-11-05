"""Endothelial tip cell that climbs the VEGF gradient and leaves a new vessel segment behind it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, VisualSpec
from ._sprout_view import SproutView


class SproutElement(Element):
    """Endothelial tip cell that climbs the VEGF gradient and leaves a new vessel segment behind it.

    Every step the tip moves to a neighbouring cell with a probability that grows with `exp(chemotaxis * VEGF /
    VEGF_here)`, so it senses the relative gradient, and it leaves a vessel segment where there was none. With
    probability `branching` the tip splits in two, the new tip standing where the tip was. Below the VEGF level
    `activation` the tip stays put. The system adds the vessel segment, the branching, and the step as member
    interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        chemotaxis:
            How strongly VEGF draws the tip.

        branching:
            The probability per step of a new tip.

        activation:
            The VEGF above which the tip moves.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.oncology import SproutElement
        >>> GroupElement('sprouts', element=SproutElement, size=2, location=(0, 5)).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        chemotaxis: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        branching: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        activation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.chemotaxis = chemotaxis
        self.branching = branching
        self.activation = activation

    @classmethod
    def _get_view(cls) -> type[SproutView]:
        """Return the view of a sprout tip, `SproutView`."""
        return SproutView
