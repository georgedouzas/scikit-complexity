"""Markets models: a commodity market whose price cycles because capacity arrives after a construction delay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import (
    Auxiliary,
    Element,
    Law,
    Sink,
    Source,
    Symbols,
    System,
    Transfer,
    delay3,
    lookup,
    smooth,
    step,
)
from ....core.runtime import register_model

PRICE_TABLE = ((0.0, 3.0), (0.5, 1.8), (1.0, 1.0), (1.5, 0.65), (2.0, 0.5), (3.0, 0.4))
INVESTMENT_TABLE = ((0.0, 0.0), (0.5, 0.1), (1.0, 1.0), (1.25, 2.5), (1.5, 4.0), (2.0, 5.0), (3.0, 5.5))


def _commodity_laws(s: Symbols) -> list[Law]:
    """Return the laws of the commodity market."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    coverage = Auxiliary('coverage', s.inventory / demand)
    price = Auxiliary('price', s.reference_price * lookup(coverage.symbol / s.desired_coverage, PRICE_TABLE))
    relative_price = price.symbol / s.reference_price
    expected = Auxiliary('expected_price', smooth(price.symbol, s.expectation_time))
    production = s.capacity * relative_price**s.supply_elasticity
    starts = Auxiliary(
        'capacity_orders',
        s.capacity / s.lifetime * lookup(expected.symbol / s.reference_price, INVESTMENT_TABLE),
    )
    return [
        coverage,
        price,
        expected,
        starts,
        Transfer(Source('producers'), s.inventory, production, label='production'),
        Transfer(
            s.inventory,
            Sink('consumers'),
            demand * relative_price ** (-s.demand_elasticity),
            label='consumption',
        ),
        Transfer(Source('construction'), s.capacity, delay3(starts.symbol, s.construction_delay), label='completions'),
        Transfer(s.capacity, Sink('retirement'), s.capacity / s.lifetime, label='retirements'),
    ]


CommodityMarket = Element.define(
    'CommodityMarket',
    states={
        'inventory': state(default=25.0, domain=Domain.NONNEGATIVE, quantity='commodity'),
        'capacity': state(default=100.0, domain=Domain.NONNEGATIVE, quantity='capacity'),
    },
    parameters={
        'demand': parameter(default=100.0, domain=Domain.POSITIVE),
        'demand_step': parameter(default=10.0, domain=Domain.REAL),
        'step_time': parameter(default=2.0, domain=Domain.NONNEGATIVE),
        'desired_coverage': parameter(default=0.25, domain=Domain.POSITIVE),
        'reference_price': parameter(default=1.0, domain=Domain.POSITIVE),
        'supply_elasticity': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'demand_elasticity': parameter(default=0.25, domain=Domain.NONNEGATIVE),
        'expectation_time': parameter(default=1.0, domain=Domain.POSITIVE),
        'construction_delay': parameter(default=3.0, domain=Domain.POSITIVE),
        'lifetime': parameter(default=15.0, domain=Domain.POSITIVE),
    },
    laws=_commodity_laws,
)


@register_model(
    field='economics/markets',
    tags=('commodity cycle', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 60.0, 'dt': 0.25},
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
            `market.production`, `market.consumption`, `market.completions`, and `market.retirements`, and the
            auxiliaries `market.coverage`, `market.price`, `market.expected_price`, and `market.capacity_orders`.

    Examples:
        >>> from skcomplex.models.economics.markets import commodity_cycle
        >>> commodity_cycle(construction_delay=5.0).elements[0].construction_delay
        5.0
    """
    market = CommodityMarket(
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
    return System(components=[market])
