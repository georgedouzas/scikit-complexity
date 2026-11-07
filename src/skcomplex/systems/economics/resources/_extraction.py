"""Natural resources systems: capital that grows by extracting a nonrenewable or a renewable resource."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.laws.utils import clip, lookup
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system

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


RESOURCE_LEVELS = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)


YIELD_LEVELS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def _table(levels: Sequence[float], values: Sequence[float]) -> list[tuple[float, float]]:
    """Return the points of a table function from its inputs and outputs."""
    return [(float(level), float(value)) for level, value in zip(levels, values, strict=True)]


def _efficiency(holder: SimpleNamespace, yields: Sequence[float]) -> Expression:
    """Return the yield per unit of capital, a table of the resource stock."""
    return lookup(holder.resource, _table(RESOURCE_LEVELS, yields))


def _investment(holder: SimpleNamespace, revenue: Expression) -> Expression:
    """Return the investment, the profit of a revenue clipped to the growth goal per unit of capital."""
    return clip(revenue - holder.cost * holder.capital, 0, holder.growth_goal * holder.capital)


def _oil_price(holder: SimpleNamespace, yields: Sequence[float], prices: Sequence[float] | None) -> Expression:
    """Return the price of the oil, constant or a table of the yield."""
    return holder.price if prices is None else lookup(_efficiency(holder, yields), _table(YIELD_LEVELS, prices))


def _harvest(holder: SimpleNamespace, yields: Sequence[float]) -> Expression:
    """Return the harvest or extraction, the capital times the yield per unit of capital."""
    return holder.capital * _efficiency(holder, yields)


def _fish_revenue(holder: SimpleNamespace, yields: Sequence[float], prices: Sequence[float]) -> Expression:
    """Return the revenue of the harvest at the price, a table of the yield."""
    return lookup(_efficiency(holder, yields), _table(YIELD_LEVELS, prices)) * _harvest(holder, yields)


def _capital(label: str, holder: str) -> TransferInteraction:
    """Return the depreciation of the capital of a holder, reported as `<holder>.depreciation`."""
    return TransferInteraction(
        label,
        source=holder,
        target=holder,
        state='capital',
        into=Sink('scrap'),
        amount=lambda s: s.source.capital / s.source.lifetime,
        flow='depreciation',
    )


@register_system(
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
            The system, with the resource `oil.resource` and the capital `oil.capital`.

    Examples:
        >>> from skcomplex.systems.economics.resources import oil_economy
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
    yields, prices = economy.yield_table, economy.price_table
    extraction = TransferInteraction(
        'extraction',
        source='oil',
        target='oil',
        state='resource',
        into=Sink('market'),
        amount=lambda s: _harvest(s.source, yields),
        flow='extraction',
    )
    investment = TransferInteraction(
        'investment',
        source='oil',
        target='oil',
        state=Source('investors'),
        into='capital',
        amount=lambda s: _investment(s.source, _oil_price(s.source, yields, prices) * _harvest(s.source, yields)),
        flow='investment',
    )
    return System(elements=[economy], interactions=[extraction, investment, _capital('depreciation', 'oil')])


@register_system(
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
            The system, with the resource `fishery.resource` and the capital `fishery.capital`.

    Examples:
        >>> from skcomplex.systems.economics.resources import fishery_capital
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
    yields, regenerations, prices = fishery.yield_table, fishery.regeneration_table, fishery.price_table
    fish = TransferInteraction(
        'fish',
        source='fishery',
        target='fishery',
        state=Source('spawning'),
        into='resource',
        amount=lambda s: s.source.resource * lookup(s.source.resource, _table(RESOURCE_LEVELS, regenerations)),
        flow='regeneration',
        legs=[('resource', Sink('market'), lambda s: _harvest(s.source, yields), 'harvest')],
    )
    investment = TransferInteraction(
        'investment',
        source='fishery',
        target='fishery',
        state=Source('investors'),
        into='capital',
        amount=lambda s: _investment(s.source, _fish_revenue(s.source, yields, prices)),
        flow='investment',
    )
    return System(elements=[fishery], interactions=[fish, investment, _capital('depreciation', 'fishery')])
