"""Market of one good described by its price."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class MarketElement(Element):
    """Market of one good described by its price.

    Args:
        label:
            The label of the market.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.markets import MarketElement
        >>> MarketElement('wheat', price=2.0).price
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
