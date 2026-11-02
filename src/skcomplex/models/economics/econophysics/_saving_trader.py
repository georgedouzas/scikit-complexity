"""Agent of the kinetic wealth exchange models, whose wealth changes only in pairwise trades."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._saving_trader_view import SavingTraderView


class SavingTraderElement(Element):
    """Agent of the kinetic wealth exchange models, whose wealth changes only in pairwise trades.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The initial wealth, negative down to the debt limit.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.econophysics import SavingTraderElement
        >>> group = GroupElement('traders', element=SavingTraderElement, size=10, values={'wealth': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('traders', 'SavingTraderElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[SavingTraderView]:
        """Return the view of a trader, with its wealth."""
        return SavingTraderView
