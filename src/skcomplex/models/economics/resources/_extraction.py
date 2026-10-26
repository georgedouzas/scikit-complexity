"""Natural resources models: capital that grows by extracting a nonrenewable or a renewable resource."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

from ....core.base import BaseLaw, Expression, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import Sink, Source, Symbols, TransferLaw, lookup, where
from ....core.system import System
from ....engine.runtime import register_model

RESOURCE_LEVELS = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
OIL_YIELD = (0.0, 0.1, 0.25, 0.45, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0, 1.0)
FISH_YIELD = (0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0)
REGENERATION = (0.0, 0.25, 0.4, 0.48, 0.5, 0.5, 0.48, 0.4, 0.3, 0.15, 0.0)
YIELD_LEVELS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
PRICE_BY_YIELD = (10.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.2)


def _table(levels: Sequence[float], values: Sequence[float]) -> list[tuple[float, float]]:
    """Return the points of a table function from its inputs and outputs."""
    return [(float(level), float(value)) for level, value in zip(levels, values, strict=True)]


def _investment(profit: Expression, goal: Expression) -> Expression:
    """Return the investment, the profit capped by the growth goal, never negative."""
    capped = where(profit < goal, profit, goal)
    return where(capped > 0, capped, 0)


class OilEconomyElement(Element):
    """Capital that extracts a nonrenewable resource and reinvests its profit up to a growth goal.

    Extraction is the capital times the yield per unit of capital, a table of the resource stock. The profit is the
    price times the extraction minus the operating cost of the capital. The price is constant, or a table of the yield
    when a price table is given.

    Args:
        label:
            The label of the economy.

        yield_table:
            The yield per unit of capital at the resource levels 0, 100, ..., 1000.

        price_table:
            The price at the yields 0, 0.1, 0.2, 0.4, 0.6, 0.8, and 1, or `None` for a constant price.

        **values:
            The values of the declared variables, by name:

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

    Examples:
        >>> from skcomplex.models.economics.resources import OilEconomyElement
        >>> OilEconomyElement('oil', price=4.0).price
        4.0
    """

    resource: float = state(default=1000.0, domain=NumericDomain.NONNEGATIVE, quantity='oil')
    capital: float = state(default=5.0, domain=NumericDomain.NONNEGATIVE, quantity='capital')
    price: float = parameter(default=3.0, domain=NumericDomain.POSITIVE)
    cost: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)
    growth_goal: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)
    lifetime: float = parameter(default=20.0, domain=NumericDomain.POSITIVE)

    def __init__(
        self,
        label: str,
        *,
        yield_table: Sequence[float] = OIL_YIELD,
        price_table: Sequence[float] | None = None,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.yield_table = yield_table
        self.price_table = price_table

    def _profit(self, s: Symbols, efficiency: Expression) -> Expression:
        """Return the profit of the extraction at a yield per unit of capital."""
        price = s.price
        if self.price_table is not None:
            price = lookup(efficiency, _table(YIELD_LEVELS, self.price_table))
        return price * s.capital * efficiency - s.cost * s.capital

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the extraction from the resource and the investment in and depreciation of the capital.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        efficiency = lookup(s.resource, _table(RESOURCE_LEVELS, self.yield_table))
        extraction = s.capital * efficiency
        investment = _investment(self._profit(s, efficiency), s.growth_goal * s.capital)
        return [
            TransferLaw(s.resource, Sink('market'), extraction, label='extraction'),
            TransferLaw(Source('investors'), s.capital, investment, label='investment'),
            TransferLaw(s.capital, Sink('scrap'), s.capital / s.lifetime, label='depreciation'),
        ]


class FisheryElement(Element):
    """Capital that harvests a regenerating resource, with a price that rises as the yield falls.

    The resource regenerates at a rate per unit of stock that is a hump-shaped table of the stock, zero when it is
    empty or full. The harvest is the capital times the yield per unit of capital, a table of the stock.

    Args:
        label:
            The label of the fishery.

        yield_table:
            The yield per unit of capital at the resource levels 0, 100, ..., 1000.

        regeneration_table:
            The regeneration rate at the resource levels 0, 100, ..., 1000.

        price_table:
            The price at the yields 0, 0.1, 0.2, 0.4, 0.6, 0.8, and 1.

        **values:
            The values of the declared variables, by name:

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

    Examples:
        >>> from skcomplex.models.economics.resources import FisheryElement
        >>> FisheryElement('fishery', lifetime=15.0).lifetime
        15.0
    """

    resource: float = state(default=1000.0, domain=NumericDomain.NONNEGATIVE, quantity='fish')
    capital: float = state(default=5.0, domain=NumericDomain.NONNEGATIVE, quantity='capital')
    cost: float = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    growth_goal: float = parameter(default=0.1, domain=NumericDomain.NONNEGATIVE)
    lifetime: float = parameter(default=20.0, domain=NumericDomain.POSITIVE)

    def __init__(
        self,
        label: str,
        *,
        yield_table: Sequence[float] = FISH_YIELD,
        regeneration_table: Sequence[float] = REGENERATION,
        price_table: Sequence[float] = PRICE_BY_YIELD,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.yield_table = yield_table
        self.regeneration_table = regeneration_table
        self.price_table = price_table

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the regeneration and harvest of the resource and the investment in and depreciation of the capital.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        efficiency = lookup(s.resource, _table(RESOURCE_LEVELS, self.yield_table))
        price = lookup(efficiency, _table(YIELD_LEVELS, self.price_table))
        harvest = s.capital * efficiency
        regeneration = s.resource * lookup(s.resource, _table(RESOURCE_LEVELS, self.regeneration_table))
        investment = _investment(price * harvest - s.cost * s.capital, s.growth_goal * s.capital)
        return [
            TransferLaw(Source('spawning'), s.resource, regeneration, label='regeneration'),
            TransferLaw(s.resource, Sink('market'), harvest, label='harvest'),
            TransferLaw(Source('investors'), s.capital, investment, label='investment'),
            TransferLaw(s.capital, Sink('scrap'), s.capital / s.lifetime, label='depreciation'),
        ]


@register_model(
    field='economics/resources',
    tags=('nonrenewable resource', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 1.0},
)
def oil_economy(
    resource: float = 1000.0,
    capital: float = 5.0,
    *,
    price: float = 3.0,
    cost: float = 0.1,
    growth_goal: float = 0.1,
    lifetime: float = 20.0,
    price_table: Sequence[float] | None = None,
) -> System:
    """Return Meadows' oil economy, capital that grows on a nonrenewable resource and collapses as it runs out.

    The capital grows exponentially while extraction is cheap, peaks as falling yields cut the profit, and declines
    with depreciation, leaving part of the resource in the ground.

    Args:
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

        price_table:
            The price at the yields 0, 0.1, 0.2, 0.4, 0.6, 0.8, and 1, or `None` for a constant price.

    Returns:
        system:
            The model, with the resource `oil.resource` and the capital `oil.capital`.

    Examples:
        >>> from skcomplex.models.economics.resources import oil_economy
        >>> oil_economy(resource=2000.0).elements[0].resource
        2000.0
    """
    economy = OilEconomyElement(
        'oil',
        resource=resource,
        capital=capital,
        price=price,
        cost=cost,
        growth_goal=growth_goal,
        lifetime=lifetime,
        price_table=None if price_table is None else tuple(float(value) for value in price_table),
    )
    return System(components=[economy])


@register_model(
    field='economics/resources',
    tags=('renewable resource', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 1.0},
)
def fishery_capital(
    resource: float = 1000.0,
    capital: float = 5.0,
    *,
    cost: float = 1.0,
    growth_goal: float = 0.1,
    lifetime: float = 20.0,
    yield_table: Sequence[float] = FISH_YIELD,
) -> System:
    """Return Meadows' fishing fleet, capital that grows on a renewable resource and can overshoot it.

    The capital grows while the harvest pays, and the outcome depends on the yield per unit of capital. With the default
    yield, which falls with the square of the stock, the fleet and the stock settle at an equilibrium. A technology that
    keeps the yield high at low stocks lets the fleet overshoot, and the stock and then the fleet collapse.

    Args:
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

    Returns:
        system:
            The model, with the resource `fishery.resource` and the capital `fishery.capital`.

    Examples:
        >>> from skcomplex.models.economics.resources import fishery_capital
        >>> fishery_capital(lifetime=10.0).elements[0].lifetime
        10.0
    """
    fishery = FisheryElement(
        'fishery',
        resource=resource,
        capital=capital,
        cost=cost,
        growth_goal=growth_goal,
        lifetime=lifetime,
        yield_table=tuple(float(value) for value in yield_table),
    )
    return System(components=[fishery])
