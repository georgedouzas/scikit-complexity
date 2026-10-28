"""Share the trader element of the tests of the economics realizations."""

from typing import Annotated

from skcomplex.core.elements import Element
from skcomplex.specs import StateSpec, VisualSpec


class TraderElement(Element):
    """Trader with prices, quantities, money, goods, and a value."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        bid: Annotated[float, StateSpec()] = 0.0,
        ask: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, StateSpec()] = 0.0,
        supply: Annotated[float, StateSpec()] = 0.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
        value: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bid = bid
        self.ask = ask
        self.demand = demand
        self.supply = supply
        self.money = money
        self.goods = goods
        self.value = value
