"""Markets models: a commodity market whose price cycles because capacity arrives after a construction delay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.laws.utils import distributed_delay, lookup, step
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

PRICE_TABLE = ((0.0, 3.0), (0.5, 1.8), (1.0, 1.0), (1.5, 0.65), (2.0, 0.5), (3.0, 0.4))
INVESTMENT_TABLE = ((0.0, 0.0), (0.5, 0.1), (1.0, 1.0), (1.25, 2.5), (1.5, 4.0), (2.0, 5.0), (3.0, 5.5))


def _demand(s: SimpleNamespace) -> Expression:
    """Return the demand at the reference price, after its step."""
    return s.demand + step(s.t, s.step_time, s.demand_step)


def _coverage(s: SimpleNamespace) -> Expression:
    """Return the coverage of the demand by the inventory."""
    return s.inventory / _demand(s)


def _price(s: SimpleNamespace) -> Expression:
    """Return the price, which falls as the coverage rises above the desired coverage."""
    return s.reference_price * lookup(_coverage(s) / s.desired_coverage, PRICE_TABLE)


def _expected_price(s: SimpleNamespace) -> Expression:
    """Return the expected price, the price smoothed over the expectation time."""
    return distributed_delay(_price(s), conserved=False, time=s.expectation_time)


def _capacity_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of capacity, which rise with the expected price."""
    return s.capacity / s.lifetime * lookup(_expected_price(s) / s.reference_price, INVESTMENT_TABLE)


def _commodity_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of the commodity market."""
    relative_price = _price(s) / s.reference_price
    production = s.capacity * relative_price**s.supply_elasticity
    return [
        TransferLaw(Source('producers'), s.inventory, production, label='production'),
        TransferLaw(
            s.inventory,
            Sink('consumers'),
            _demand(s) * relative_price ** (-s.demand_elasticity),
            label='consumption',
        ),
        TransferLaw(
            Source('construction'),
            s.capacity,
            distributed_delay(_capacity_orders(s), order=3, time=s.construction_delay),
            label='completions',
        ),
        TransferLaw(s.capacity, Sink('retirement'), s.capacity / s.lifetime, label='retirements'),
    ]


class CommodityMarketElement(Element):
    """Commodity market element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        inventory: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='commodity')] = 25.0,
        capacity: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='capacity')] = 100.0,
        demand: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        demand_step: Annotated[object, ParameterSpec()] = 10.0,
        step_time: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        desired_coverage: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
        reference_price: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        supply_elasticity: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        demand_elasticity: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        expectation_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        construction_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        lifetime: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the commodity market element."""
        return _commodity_laws(s)


@register_model(
    field='economics/markets',
    tags=('commodity cycle', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 60.0, 'dt': 0.25},
    observables=lambda system: [
        ObservableSpec('coverage', _coverage, element='market'),
        ObservableSpec('price', _price, element='market'),
        ObservableSpec('expected_price', _expected_price, element='market'),
        ObservableSpec('capacity_orders', _capacity_orders, element='market'),
    ],
)
def commodity_cycle(
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
) -> System:
    """Return a commodity market whose price and capacity cycle because new capacity arrives after a delay.

    The inventory is a stock of the commodity that production fills and consumption drains. The price is a falling
    table function of the inventory coverage, the years of demand the inventory holds, relative to the desired
    coverage. A high price raises the utilization of capacity and lowers consumption with constant elasticities.
    Producers form an expected price by smoothing the price, and order new capacity at the rate of retirements
    multiplied by a rising table function of the expected price. Orders become capacity through a third-order delay.
    After a step in demand the market overshoots, so the price and the capacity oscillate, the commodity cycle of
    Bala et al. Section 4.11.4. The book was not at hand, so the tables and the parameters are a documented
    reconstruction of that structure, in years and in units of the reference demand and price. The market starts in
    equilibrium.

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

    Returns:
        system:
            The model, with the inventory `market.inventory` and the capacity `market.capacity` in years, the flows
            `market.production`, `market.consumption`, `market.completions`, and `market.retirements`. The observables
            of the model are `market.coverage`, `market.price`, `market.expected_price`, and `market.capacity_orders`.

    Examples:
        >>> from skcomplex.models.economics.markets import commodity_cycle
        >>> commodity_cycle(construction_delay=5.0).elements[0].construction_delay
        5.0
    """
    market = CommodityMarketElement(
        'market',
        inventory=inventory,
        capacity=capacity,
        demand=demand,
        demand_step=demand_step,
        step_time=step_time,
        desired_coverage=desired_coverage,
        reference_price=reference_price,
        supply_elasticity=supply_elasticity,
        demand_elasticity=demand_elasticity,
        expectation_time=expectation_time,
        construction_delay=construction_delay,
        lifetime=lifetime,
    )
    return System(elements=[market])
