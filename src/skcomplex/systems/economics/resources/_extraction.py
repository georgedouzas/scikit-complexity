"""Natural resources systems: capital that grows by extracting a resource.

The resource is nonrenewable or renewable.
"""

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
from ....core.components.laws.utils import clip, series
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

FISH_YIELD = (0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0)


REGENERATION = (0.0, 0.25, 0.4, 0.48, 0.5, 0.5, 0.48, 0.4, 0.3, 0.15, 0.0)


PRICE_BY_YIELD = (10.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.2)


class FisheryElement(Element):
    """Capital that harvests a regenerating resource.

    Its price rises as the yield falls. The resource regenerates at a rate per unit of stock that is a hump-shaped table
    of the stock, zero when it is empty or full. The harvest is the capital times the yield per unit of capital, a table
    of the stock. The fishery holds the stocks, the parameters, and the tables, and transfers within it of the system
    move the resource and the capital.

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
    """Capital that extracts a nonrenewable resource.

    It reinvests its profit up to a growth goal. Extraction is the capital times the yield per unit of capital, a table
    of the resource stock. The profit is the price times the extraction minus the operating cost of the capital. The
    price is constant, or a table of the yield when a price table is given. The economy holds the stocks, the
    parameters, and the tables, and transfers within it of the system move the resource and the capital.

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


def _efficiency(holder: SimpleNamespace, yields: Sequence[float]) -> Expression:
    """Return the yield per unit of capital, a table of the resource stock."""
    return series(holder.resource, RESOURCE_LEVELS, yields)


def _investment(holder: SimpleNamespace, revenue: Expression) -> Expression:
    """Return the investment.

    It is the profit of a revenue clipped to the growth goal per unit of capital.
    """
    return clip(revenue - holder.cost * holder.capital, 0, holder.growth_goal * holder.capital)


def _harvest(holder: SimpleNamespace, yields: Sequence[float]) -> Expression:
    """Return the harvest or extraction.

    It is the capital times the yield per unit of capital.
    """
    return holder.capital * _efficiency(holder, yields)


def _capital(label: str, holder: str) -> TransferInteraction:
    """Return the depreciation of the capital of a holder.

    It is reported as `<holder>.depreciation`.
    """
    return TransferInteraction(
        label,
        source=holder,
        target=holder,
        state='capital',
        into=Sink('scrap'),
        amount=lambda s: s.source.capital / s.source.lifetime,
        flow='depreciation',
    )


class OilEconomySystem(BaseSystem):
    """Meadows' oil economy, capital that grows on a nonrenewable resource.

    The capital collapses as the resource runs out. The capital grows exponentially while extraction is cheap, peaks as
    falling yields cut the profit, and declines with depreciation, leaving part of the resource in the ground. The
    system has the resource `oil.resource` and the capital `oil.capital`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import OilEconomySystem
        >>> OilEconomySystem(resource=2000.0).prepare().system_.elements[0].resource
        2000.0
    """

    field = 'economics/resources'
    topics = ('nonrenewable resource', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        resource: float = 1000.0,
        capital: float = 5.0,
        *,
        price: float = 3.0,
        cost: float = 0.1,
        growth_goal: float = 0.1,
        lifetime: float = 20.0,
        price_table: Sequence[float] | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.resource = resource
        self.capital = capital
        self.price = price
        self.cost = cost
        self.growth_goal = growth_goal
        self.lifetime = lifetime
        self.price_table = price_table

    @staticmethod
    def _oil_price(holder: SimpleNamespace, yields: Sequence[float], prices: Sequence[float] | None) -> Expression:
        """Return the price of the oil, constant or a table of the yield."""
        return holder.price if prices is None else series(_efficiency(holder, yields), YIELD_LEVELS, prices)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        economy = OilEconomyElement(
            'oil',
            resource=self.resource,
            capital=self.capital,
            price=self.price,
            cost=self.cost,
            growth_goal=self.growth_goal,
            lifetime=self.lifetime,
            price_table=None if self.price_table is None else tuple(float(value) for value in self.price_table),
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
            amount=lambda s: _investment(
                s.source,
                self._oil_price(s.source, yields, prices) * _harvest(s.source, yields),
            ),
            flow='investment',
        )
        return System(
            self.label,
            elements=[economy],
            interactions=[extraction, investment, _capital('depreciation', 'oil')],
        )


class FisheryCapitalSystem(BaseSystem):
    """Meadows' fishing fleet, capital that grows on a renewable resource.

    The capital can overshoot the resource. The capital grows while the harvest pays, and the outcome depends on the
    yield per unit of capital. With the default yield, which falls with the square of the stock, the fleet and the stock
    settle at an equilibrium. A technology that keeps the yield high at low stocks lets the fleet overshoot, and the
    stock and then the fleet collapse. The system has the resource `fishery.resource` and the capital `fishery.capital`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import FisheryCapitalSystem
        >>> FisheryCapitalSystem(lifetime=10.0).prepare().system_.elements[0].lifetime
        10.0
    """

    field = 'economics/resources'
    topics = ('renewable resource', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        resource: float = 1000.0,
        capital: float = 5.0,
        *,
        cost: float = 1.0,
        growth_goal: float = 0.1,
        lifetime: float = 20.0,
        yield_table: Sequence[float] = FISH_YIELD,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.resource = resource
        self.capital = capital
        self.cost = cost
        self.growth_goal = growth_goal
        self.lifetime = lifetime
        self.yield_table = yield_table

    @staticmethod
    def _fish_revenue(holder: SimpleNamespace, yields: Sequence[float], prices: Sequence[float]) -> Expression:
        """Return the revenue of the harvest at the price, a table of the yield."""
        return series(_efficiency(holder, yields), YIELD_LEVELS, prices) * _harvest(holder, yields)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        fishery = FisheryElement(
            'fishery',
            resource=self.resource,
            capital=self.capital,
            cost=self.cost,
            growth_goal=self.growth_goal,
            lifetime=self.lifetime,
            yield_table=tuple(float(value) for value in self.yield_table),
        )
        yields, regenerations, prices = fishery.yield_table, fishery.regeneration_table, fishery.price_table
        fish = TransferInteraction(
            'fish',
            source='fishery',
            target='fishery',
            state=Source('spawning'),
            into='resource',
            amount=lambda s: s.source.resource * series(s.source.resource, RESOURCE_LEVELS, regenerations),
            flow='regeneration',
            legs=[('resource', Sink('market'), lambda s: _harvest(s.source, yields), 'harvest')],
        )
        investment = TransferInteraction(
            'investment',
            source='fishery',
            target='fishery',
            state=Source('investors'),
            into='capital',
            amount=lambda s: _investment(s.source, self._fish_revenue(s.source, yields, prices)),
            flow='investment',
        )
        return System(
            self.label,
            elements=[fishery],
            interactions=[fish, investment, _capital('depreciation', 'fishery')],
        )
