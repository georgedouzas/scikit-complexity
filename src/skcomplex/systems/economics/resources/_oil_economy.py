"""Capital that extracts a nonrenewable resource and reinvests its profit up to a growth goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

OIL_YIELD = (0.0, 0.1, 0.25, 0.45, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0, 1.0)


class OilEconomyElement(Element):
    """Capital that extracts a nonrenewable resource and reinvests its profit up to a growth goal.

    Extraction is the capital times the yield per unit of capital, a table of the resource stock. The profit is the
    price times the extraction minus the operating cost of the capital. The price is constant, or a table of the yield
    when a price table is given. The economy holds the stocks, the parameters, and the tables, and transfers within it
    of the system move the resource and the capital.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        resource:
            The initial resource stock.

        capital:
            The initial capital.

        price:
            The price of the resource, used without a price table.

        cost:
            The operating cost per unit of capital.

        growth_goal:
            The largest investment per unit of capital.

        lifetime:
            The average lifetime of the capital.

        yield_table:
            The yield per unit of capital at the resource levels 0, 100, ..., 1000.

        price_table:
            The price at the yields 0, 0.1, 0.2, 0.4, 0.6, 0.8, and 1, or `None` for a constant price.

    Examples:
        >>> from skcomplex.systems.economics.resources import OilEconomyElement
        >>> OilEconomyElement('oil', price=4.0).price
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        resource: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='oil')] = 1000.0,
        capital: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='capital')] = 5.0,
        price: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        growth_goal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lifetime: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        yield_table: Sequence[float] = OIL_YIELD,
        price_table: Sequence[float] | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.resource = resource
        self.capital = capital
        self.price = price
        self.cost = cost
        self.growth_goal = growth_goal
        self.lifetime = lifetime
        self.yield_table = yield_table
        self.price_table = price_table
