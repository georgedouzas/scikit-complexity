"""Agent that moves to the richest free cell in sight, harvests its sugar, burns its metabolism, and may starve."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._forager_view import ForagerView


class ForagerElement(Element):
    """Agent that moves to the richest free cell in sight, harvests its sugar, burns its metabolism, and may starve.

    In its own activation, a forager sees `vision` cells along each of the four lattice directions and moves to the
    free cell with the most sugar, the nearest among equals, or stays when no cell in sight beats its own, by a
    `MovementInteraction` of the best step with a vision. Transfers then harvest the sugar of its cell and burn its
    metabolism, and a `LifeCycleInteraction` removes it when its wealth falls below zero.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The wealth of the forager.

        vision:
            The number of cells the forager sees in every direction.

        metabolism:
            The sugar the forager burns in a step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import ForagerElement
        >>> GroupElement('foragers', element=ForagerElement, size=3, values={'vision': 2.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar'}),
        *,
        sugar: Annotated[float, StateSpec()] = 10.0,
        vision: Annotated[float, StateSpec(domain=NN)] = 1.0,
        metabolism: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.vision = vision
        self.metabolism = metabolism

    @classmethod
    def _get_view(cls) -> type[ForagerView]:
        """Return the view of a forager, with its sugar, vision, and metabolism."""
        return ForagerView
