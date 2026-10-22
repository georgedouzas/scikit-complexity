"""Markets models: a retailer's inventory under perception, response, and delivery delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Auxiliary, Law, Sink, Source, Symbols, Transfer, distributed_delay, step, where
from ....core.runtime import register_model
from ....core.system import System


def _dealer_laws(s: Symbols) -> list[Law]:
    """Return the laws of the car dealership."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    sales = where(s.inventory < demand, s.inventory, demand)
    perceived = Auxiliary('perceived', distributed_delay(sales, conserved=False, time=s.perception_delay))
    desired = Auxiliary('desired_inventory', s.coverage * perceived.symbol)
    gap_orders = perceived.symbol + (desired.symbol - s.inventory) / s.response_delay
    orders = Auxiliary('orders', where(gap_orders > 0, gap_orders, 0))
    return [
        perceived,
        desired,
        orders,
        Transfer(
            Source('factory'),
            s.inventory,
            distributed_delay(orders.symbol, order=3, time=s.delivery_delay),
            label='deliveries',
        ),
        Transfer(s.inventory, Sink('customers'), sales, label='sales'),
    ]


CarDealership = Element.define(
    'CarDealership',
    states={'inventory': state(default=200.0, domain=Domain.NONNEGATIVE, quantity='cars')},
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
    tags=('inventory', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode'),
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

    The inventory is a stock of cars that deliveries from the factory fill and sales to customers drain. Sales are the
    customer demand, limited by the cars on the lot. The dealer smooths sales over the perception delay, wants an
    inventory of `coverage` days of perceived sales, and orders the perceived sales plus the gap to that inventory
    divided by the response delay. Orders arrive through a third-order material delay, a continuous stand-in for
    Meadows' fixed delivery delay. Lengthening the response delay damps the oscillation, while shortening the
    perception delay makes the dealer react faster to the new demand and slightly enlarges it.

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
            The average time between an order and its delivery.

    Returns:
        system:
            The model, with the cars on the lot `dealer.inventory`, the flows `dealer.deliveries` and `dealer.sales`,
            and the auxiliaries `dealer.perceived` for the perceived sales, `dealer.desired_inventory`, and
            `dealer.orders`.

    Examples:
        >>> from skcomplex.models.economics.markets import car_dealer
        >>> car_dealer(response_delay=6.0).elements[0].response_delay
        6.0
    """
    dealer = CarDealership(
        'dealer',
        inventory=inventory,
        demand=demand,
        demand_step=demand_step,
        step_time=step_time,
        coverage=coverage,
        perception_delay=perception_delay,
        response_delay=response_delay,
        delivery_delay=delivery_delay,
    )
    return System(components=[dealer])
