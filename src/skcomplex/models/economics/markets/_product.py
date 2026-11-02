"""Product on the shelf of a shop, which ages from the time it was shelved, its freshness `exp(-age / tau_a)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._customer_view import CustomerView
from ._product_view import ProductView


class ProductElement(Element):
    """Product on the shelf of a shop, which ages from the time it was shelved, its freshness `exp(-age / tau_a)`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        shelved:
            The time at which the product was put on the shelf.

        tau_a:
            The time over which the product ages.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import ProductElement
        >>> GroupElement('products', element=ProductElement, size=4, location=[0, 0, 1, 1]).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        shelved: Annotated[float, StateSpec()] = 0.0,
        tau_a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.shelved = shelved
        self.tau_a = tau_a

    @classmethod
    def _get_view(cls) -> type[ProductView]:
        """Return the view of a product, with its shelving time and ageing time."""
        return ProductView

    @staticmethod
    def restock(product: ProductView, customer: CustomerView, model: ModelView) -> float:
        """Return the amount `shelved - t` that resets the shelving time of a sold product to the time of the sale.

        Args:
            product:
                The product sold.

            customer:
                The customer who buys it.

            model:
                The model view of the purchase.

        Returns:
            amount:
                The amount taken from the shelving time.
        """
        return product.shelved - model.time

    @staticmethod
    def sell(product: ProductView, customer: CustomerView, model: ModelView) -> float:
        """Return the amount that sets the freshness the customer holds to `exp(-(t - shelved) / tau_a)`.

        Args:
            product:
                The product sold.

            customer:
                The customer who buys it.

            model:
                The model view of the purchase.

        Returns:
            amount:
                The change of the freshness the customer holds.
        """
        return float(np.exp(-(model.time - product.shelved) / product.tau_a)) - customer.freshness
