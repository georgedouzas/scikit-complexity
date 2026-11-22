"""Markets systems: a commodity market whose price cycles.

The price cycles because capacity arrives after a construction delay.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Quantity
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import distributed_delay, series, step
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System

PRICE_TABLE = ((0.0, 0.5, 1.0, 1.5, 2.0, 3.0), (3.0, 1.8, 1.0, 0.65, 0.5, 0.4))
INVESTMENT_TABLE = ((0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0), (0.0, 0.1, 1.0, 2.5, 4.0, 5.0, 5.5))


class CommodityMarketElement(Element):
    """Commodity market whose inventory production fills and consumption drains.

    Its capacity is built after a delay. The price is a falling table function of the coverage of the demand by the
    inventory relative to the desired coverage. Production is the capacity times the relative price to the supply
    elasticity, consumption the demand times the relative price to minus the demand elasticity. Capacity is ordered at
    the rate of retirements times a rising table function of the smoothed price, completed through a third-order delay,
    and retired over its lifetime. The market holds the stocks and the parameters, and labelled flows of the system move
    the commodity and the capacity.

    Args:
        label:
            The label of the market.

        inventory:
            The initial inventory of the commodity.

        capacity:
            The initial production capacity per year.

        demand:
            The demand per year at the reference price.

        demand_step:
            The permanent change of the demand.

        step_time:
            The year of the change of the demand.

        desired_coverage:
            The years of demand producers want the inventory to hold.

        reference_price:
            The price at the desired coverage.

        supply_elasticity:
            The elasticity of the utilization of capacity with respect to the price.

        demand_elasticity:
            The elasticity of consumption with respect to the price, as a positive number.

        expectation_time:
            The time over which producers smooth the price into their expected price.

        construction_delay:
            The average time between an order of capacity and its completion.

        lifetime:
            The average lifetime of capacity.

    Examples:
        >>> from skcomplex.systems.economics.markets import CommodityMarketElement
        >>> CommodityMarketElement('market', lifetime=20.0).lifetime
        20.0
    """

    def __init__(
        self,
        label: str,
        *,
        inventory: Annotated[float, RealSet.closed_open(0, oo), Quantity('commodity')] = 25.0,
        capacity: Annotated[float, RealSet.closed_open(0, oo), Quantity('capacity')] = 100.0,
        demand: Annotated[float, RealSet.open(0, oo)] = 100.0,
        demand_step: float = 10.0,
        step_time: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        desired_coverage: Annotated[float, RealSet.open(0, oo)] = 0.25,
        reference_price: Annotated[float, RealSet.open(0, oo)] = 1.0,
        supply_elasticity: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        demand_elasticity: Annotated[float, RealSet.closed_open(0, oo)] = 0.25,
        expectation_time: Annotated[float, RealSet.open(0, oo)] = 1.0,
        construction_delay: Annotated[float, RealSet.open(0, oo)] = 3.0,
        lifetime: Annotated[float, RealSet.open(0, oo)] = 15.0,
    ) -> None:
        super().__init__(label)
        self.inventory = inventory
        self.capacity = capacity
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time
        self.desired_coverage = desired_coverage
        self.reference_price = reference_price
        self.supply_elasticity = supply_elasticity
        self.demand_elasticity = demand_elasticity
        self.expectation_time = expectation_time
        self.construction_delay = construction_delay
        self.lifetime = lifetime


def _get_coverage(s: SimpleNamespace) -> Expression:
    """Return the coverage of the demand by the inventory."""
    return s.inventory / _get_demand(s)


def _get_price(s: SimpleNamespace) -> Expression:
    """Return the price, which falls as the coverage rises above the desired coverage."""
    return s.reference_price * series(_get_coverage(s) / s.desired_coverage, *PRICE_TABLE)


def _get_expected_price(s: SimpleNamespace) -> Expression:
    """Return the expected price, the price smoothed over the expectation time."""
    return distributed_delay(_get_price(s), conserved=False, time=s.expectation_time)


def _get_capacity_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of capacity, which rise with the expected price."""
    return s.capacity / s.lifetime * series(_get_expected_price(s) / s.reference_price, *INVESTMENT_TABLE)


def _get_demand(s: SimpleNamespace) -> Expression:
    """Return the demand at the reference price, after its step."""
    return s.demand + step(s.t, s.step_time, s.demand_step)


class CommodityCycleSystem(BaseSystem):
    """A commodity market whose price and capacity cycle.

    New capacity arrives after a delay. The inventory is a stock of the commodity that production fills and consumption
    drains. The price is a falling table function of the inventory coverage, the years of demand the inventory holds,
    relative to the desired coverage. A high price raises the utilization of capacity and lowers consumption with
    constant elasticities. Producers form an expected price by smoothing the price, and order new capacity at the rate
    of retirements multiplied by a rising table function of the expected price. Orders become capacity through a
    third-order delay. After a step in demand the market overshoots, so the price and the capacity oscillate, the
    commodity cycle of Bala et al. Section 4.11.4. The book was not at hand, so the tables and the parameters are a
    documented reconstruction of that structure, in years and in units of the reference demand and price. The market
    starts in equilibrium.

    The system has the inventory `market.inventory` and the capacity `market.capacity` in years, the flows
    `market.production`, `market.consumption`, `market.completions`, and `market.retirements`. The observables of the
    system are `market.coverage`, `market.price`, `market.expected_price`, and `market.capacity_orders`.

    Args:
        inventory:
            The initial inventory of the commodity.

        capacity:
            The initial production capacity per year.

        demand:
            The demand per year at the reference price.

        demand_step:
            The permanent change of the demand.

        step_time:
            The year of the change of the demand.

        desired_coverage:
            The years of demand producers want the inventory to hold.

        reference_price:
            The price at the desired coverage.

        supply_elasticity:
            The elasticity of the utilization of capacity with respect to the price.

        demand_elasticity:
            The elasticity of consumption with respect to the price, as a positive number.

        expectation_time:
            The time over which producers smooth the price into their expected price.

        construction_delay:
            The average time between an order of capacity and its completion.

        lifetime:
            The average lifetime of capacity.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import CommodityCycleSystem
        >>> CommodityCycleSystem(construction_delay=5.0).prepare().system_.elements[0].construction_delay
        5.0
    """

    field = 'economics/markets'
    topics = ('commodity cycle', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 60.0
    dt = 0.25
    observables = (
        Observable('coverage', _get_coverage, element='market'),
        Observable('price', _get_price, element='market'),
        Observable('expected_price', _get_expected_price, element='market'),
        Observable('capacity_orders', _get_capacity_orders, element='market'),
    )

    @staticmethod
    def _get_market(s: SimpleNamespace) -> SimpleNamespace:
        """Return the symbols of the market whose commodity or capacity a flow moves, with the time."""
        return SimpleNamespace(t=s.t, **vars(s.target))

    def __init__(
        self,
        inventory: float = 25.0,
        capacity: float = 100.0,
        *,
        demand: float = 100.0,
        demand_step: float = 10.0,
        step_time: float = 2.0,
        desired_coverage: float = 0.25,
        reference_price: float = 1.0,
        supply_elasticity: float = 0.1,
        demand_elasticity: float = 0.25,
        expectation_time: float = 1.0,
        construction_delay: float = 3.0,
        lifetime: float = 15.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.inventory = inventory
        self.capacity = capacity
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time
        self.desired_coverage = desired_coverage
        self.reference_price = reference_price
        self.supply_elasticity = supply_elasticity
        self.demand_elasticity = demand_elasticity
        self.expectation_time = expectation_time
        self.construction_delay = construction_delay
        self.lifetime = lifetime

    @staticmethod
    def _get_production(s: SimpleNamespace) -> Expression:
        """Return the production.

        It is the capacity times the relative price to the supply elasticity.
        """
        market = CommodityCycleSystem._get_market(s)
        return market.capacity * (_get_price(market) / market.reference_price) ** market.supply_elasticity

    @staticmethod
    def _get_consumption(s: SimpleNamespace) -> Expression:
        """Return the consumption.

        It is the demand times the relative price to minus the demand elasticity.
        """
        market = CommodityCycleSystem._get_market(s)
        return _get_demand(market) * (_get_price(market) / market.reference_price) ** (-market.demand_elasticity)

    @staticmethod
    def _get_completions(s: SimpleNamespace) -> Expression:
        """Return the completions of capacity, the orders through a third-order delay."""
        market = CommodityCycleSystem._get_market(s)
        return distributed_delay(_get_capacity_orders(market), order=3, time=market.construction_delay)

    @staticmethod
    def _get_commodity_flows(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the production into the inventory and the consumption out of it."""
        inventory = s.target.inventory
        return [
            RateLaw({inventory: +1}, CommodityCycleSystem._get_production(s), label='production'),
            RateLaw({inventory: -1}, CommodityCycleSystem._get_consumption(s), label='consumption'),
        ]

    @staticmethod
    def _get_capacity_flows(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the completions of capacity and its retirements over its lifetime."""
        capacity = s.target.capacity
        return [
            RateLaw({capacity: +1}, CommodityCycleSystem._get_completions(s), label='completions'),
            RateLaw({capacity: -1}, capacity / s.target.lifetime, label='retirements'),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        market = CommodityMarketElement(
            'market',
            inventory=self.inventory,
            capacity=self.capacity,
            demand=self.demand,
            demand_step=self.demand_step,
            step_time=self.step_time,
            desired_coverage=self.desired_coverage,
            reference_price=self.reference_price,
            supply_elasticity=self.supply_elasticity,
            demand_elasticity=self.demand_elasticity,
            expectation_time=self.expectation_time,
            construction_delay=self.construction_delay,
            lifetime=self.lifetime,
        )
        commodity = Interaction('commodity', roles={'target': 'market'}, laws=self._get_commodity_flows)
        capacity_flows = Interaction('capacity_flows', roles={'target': 'market'}, laws=self._get_capacity_flows)
        self.system_ = System(self.label, elements=[market], interactions=[commodity, capacity_flows])
