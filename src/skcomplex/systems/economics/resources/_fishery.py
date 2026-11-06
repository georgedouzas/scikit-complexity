"""Capital that harvests a regenerating resource, with a price that rises as the yield falls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

FISH_YIELD = (0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0)


REGENERATION = (0.0, 0.25, 0.4, 0.48, 0.5, 0.5, 0.48, 0.4, 0.3, 0.15, 0.0)


PRICE_BY_YIELD = (10.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.2)


class FisheryElement(Element):
    """Capital that harvests a regenerating resource, with a price that rises as the yield falls.

    The resource regenerates at a rate per unit of stock that is a hump-shaped table of the stock, zero when it is
    empty or full. The harvest is the capital times the yield per unit of capital, a table of the stock. The fishery
    holds the stocks, the parameters, and the tables, and transfers within it of the system move the resource and the
    capital.

    Args:
        label:
            The label of the fishery.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        resource:
            The initial resource stock.

        capital:
            The initial capital.

        cost:
            The operating cost per unit of capital.

        growth_goal:
            The largest investment per unit of capital.

        lifetime:
            The average lifetime of the capital.

        yield_table:
            The yield per unit of capital at the resource levels 0, 100, ..., 1000.

        regeneration_table:
            The regeneration rate at the resource levels 0, 100, ..., 1000.

        price_table:
            The price at the yields 0, 0.1, 0.2, 0.4, 0.6, 0.8, and 1.

    Examples:
        >>> from skcomplex.systems.economics.resources import FisheryElement
        >>> FisheryElement('fishery', lifetime=15.0).lifetime
        15.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        resource: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 1000.0,
        capital: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='capital')] = 5.0,
        cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        growth_goal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lifetime: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        yield_table: Sequence[float] = FISH_YIELD,
        regeneration_table: Sequence[float] = REGENERATION,
        price_table: Sequence[float] = PRICE_BY_YIELD,
    ) -> None:
        super().__init__(label, visual=visual)
        self.resource = resource
        self.capital = capital
        self.cost = cost
        self.growth_goal = growth_goal
        self.lifetime = lifetime
        self.yield_table = yield_table
        self.regeneration_table = regeneration_table
        self.price_table = price_table
