"""Trader who holds goods and values them with Cobb-Douglas weights."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class GoodsTraderElement(Element):
    """Trader who holds goods and values them with Cobb-Douglas weights.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The holding of sugar.

        spice:
            The holding of spice.

        salt:
            The holding of salt.

        sugar_weight:
            The Cobb-Douglas weight of sugar.

        spice_weight:
            The Cobb-Douglas weight of spice.

        salt_weight:
            The Cobb-Douglas weight of salt, zero when salt is not traded.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import GoodsTraderElement
        >>> GroupElement('traders', element=GoodsTraderElement, size=4, values={'sugar': 5.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar_weight', 'color_range': (0.0, 1.0)}),
        *,
        sugar: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        spice: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        salt: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        sugar_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        spice_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        salt_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.spice = spice
        self.salt = salt
        self.sugar_weight = sugar_weight
        self.spice_weight = spice_weight
        self.salt_weight = salt_weight
