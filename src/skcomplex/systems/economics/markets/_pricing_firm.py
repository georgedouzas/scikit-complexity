"""A firm of a discretised pricing game that plays against the prices of its rival."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.components.elements import Element
from ....core.specs import StateSpec
from ._pricing_firm_view import PricingFirmView


class PricingFirmElement(Element):
    """Firm that charges one price of a grid and counts what every price would have earned.

    Args:
        label:
            The label of the firm or the group.

        price:
            The price the firm charges.

        scores:
            The sum of the payoffs of every price of the grid against the prices of the rival so far.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import PricingFirmElement
        >>> GroupElement('firms', element=PricingFirmElement, size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        price: Annotated[float, StateSpec()] = 1.0,
        scores: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.price = price
        self.scores = scores

    @classmethod
    def _get_view(cls) -> type[PricingFirmView]:
        """Return the view of the firms, which types their states."""
        return PricingFirmView
