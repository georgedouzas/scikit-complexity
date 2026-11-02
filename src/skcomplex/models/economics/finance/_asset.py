"""Traded asset described by its price."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class AssetElement(Element):
    """Traded asset described by its price.

    Args:
        label:
            The label of the asset.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.models.economics.finance import AssetElement
        >>> AssetElement('stock', price=100.0).price
        100.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
