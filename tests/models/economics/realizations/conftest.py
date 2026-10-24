"""Share the trader element of the tests of the economics realizations."""

from skcomplex.core.base import state
from skcomplex.core.components import Element


class Trader(Element):
    """Trader with prices, quantities, money, goods, and a value."""

    bid: float = state(0.0)
    ask: float = state(0.0)
    demand: float = state(0.0)
    supply: float = state(0.0)
    money: float = state(0.0)
    goods: float = state(0.0)
    value: float = state(0.0)
