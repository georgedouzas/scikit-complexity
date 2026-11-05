"""Patch of land whose food grows back by a share of its capacity every step, harvested against a resistance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._plot_view import PlotView


class PlotElement(Element):
    """Patch of land whose food grows back by a share of its capacity every step, harvested against a resistance.

    The regrowth is a `RegrowthInteraction` of the group of patches, relative to the capacity.

    Args:
        label:
            The label of the patch or the site group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        food:
            The food on the patch.

        capacity:
            The most food the patch holds.

        resistance:
            The labour lost to the resistance of the patch before any food is harvested.

        regrowth:
            The share of the capacity that grows back in a step.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import PlotElement
        >>> PlotElement('patch').regrowth
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        resistance: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food
        self.capacity = capacity
        self.resistance = resistance
        self.regrowth = regrowth

    @classmethod
    def _get_view(cls) -> type[PlotView]:
        """Return the view of a patch, with its food, capacity, resistance, and regrowth."""
        return PlotView
