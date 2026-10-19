"""Markets models: a retailer's inventory under perception, response, and delivery delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System, delayed, step, where
from ....core.runtime import register_model


def _orders(
    perceived: Expression,
    inventory: Expression,
    coverage: Expression,
    response_delay: Expression,
) -> Expression:
    """Return the orders to the factory, the perceived sales plus a share of the inventory gap, never negative."""
    orders = perceived + (coverage * perceived - inventory) / response_delay
    return where(orders > 0, orders, 0)


def _dealer_laws(s: Symbols) -> list[Law]:
    """Return the laws of the car dealership."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    sales = where(s.inventory < demand, s.inventory, demand)
    deliveries = _orders(
        delayed(s.perceived, s.delivery_delay),
        delayed(s.inventory, s.delivery_delay),
        s.coverage,
        s.response_delay,
    )
    return [Rate(s.inventory, deliveries - sales), Rate(s.perceived, (sales - s.perceived) / s.perception_delay)]


CarDealership = Element.define(
    'CarDealership',
    states={
        'inventory': state(default=200.0, domain=Domain.NONNEGATIVE),
        'perceived': state(default=20.0, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'demand': parameter(default=20.0, domain=Domain.NONNEGATIVE),
        'demand_step': parameter(default=2.0, domain=Domain.REAL),
        'step_time': parameter(default=25.0, domain=Domain.NONNEGATIVE),
        'coverage': parameter(default=10.0, domain=Domain.POSITIVE),
        'perception_delay': parameter(default=5.0, domain=Domain.POSITIVE),
        'response_delay': parameter(default=3.0, domain=Domain.POSITIVE),
        'delivery_delay': parameter(default=5.0, domain=Domain.POSITIVE),
    },
    laws=_dealer_laws,
)


@register_model(
    field='economics/markets',
    tags=('inventory', 'system dynamics', 'delay', 'oscillations'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 1.0},
)
def car_dealer(
    inventory: float = 200.0,
    *,
    demand: float = 20.0,
    demand_step: float = 2.0,
    step_time: float = 25.0,
    coverage: float = 10.0,
    perception_delay: float = 5.0,
    response_delay: float = 3.0,
    delivery_delay: float = 5.0,
) -> System:
    """Return Meadows' car dealership, whose inventory oscillates after a step in demand because of its delays.

    Sales are the customer demand, limited by the cars on the lot. The dealer smooths sales over the perception delay,
    wants an inventory of `coverage` days of perceived sales, and orders the perceived sales plus the gap to that
    inventory divided by the response delay. Orders arrive after the delivery delay.

    Args:
        inventory:
            The initial number of cars on the lot.

        demand:
            The customer demand in cars per day.

        demand_step:
            The permanent change of the demand.

        step_time:
            The day of the change of the demand.

        coverage:
            The days of perceived sales the dealer wants on the lot.

        perception_delay:
            The time over which the dealer averages sales.

        response_delay:
            The time over which the dealer closes the inventory gap.

        delivery_delay:
            The time between an order and its delivery.

    Returns:
        system:
            The model, with the cars on the lot `dealer.inventory` and the perceived sales `dealer.perceived`.

    Examples:
        >>> from skcomplex.models.economics.markets import car_dealer
        >>> car_dealer(response_delay=6.0).elements[0].response_delay
        6.0
    """
    dealer = CarDealership(
        'dealer',
        inventory=inventory,
        perceived=demand,
        demand=demand,
        demand_step=demand_step,
        step_time=step_time,
        coverage=coverage,
        perception_delay=perception_delay,
        response_delay=response_delay,
        delivery_delay=delivery_delay,
    )
    return System(elements=[dealer])
