"""Markets models: a retailer's inventory under perception, response, and delivery delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import BaseLaw, Expression, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import Sink, Source, Symbols, TransferLaw, distributed_delay, step, where
from ....core.system import System, observe_element
from ....engine.runtime import register_model


def _sales(s: Symbols) -> Expression:
    """Return the sales of the car dealership, the demand up to the inventory."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    return where(s.inventory < demand, s.inventory, demand)


def _perceived(s: Symbols) -> Expression:
    """Return the perceived sales of the car dealership, its sales smoothed over the perception delay."""
    return distributed_delay(_sales(s), conserved=False, time=s.perception_delay)


def _desired_inventory(s: Symbols) -> Expression:
    """Return the inventory the car dealership desires, a coverage of its perceived sales."""
    return s.coverage * _perceived(s)


def _orders(s: Symbols) -> Expression:
    """Return the orders of the car dealership, its perceived sales and the gap to its desired inventory."""
    gap_orders = _perceived(s) + (_desired_inventory(s) - s.inventory) / s.response_delay
    return where(gap_orders > 0, gap_orders, 0)


def _dealer_laws(s: Symbols) -> list[BaseLaw]:
    """Return the laws of the car dealership."""
    return [
        TransferLaw(
            Source('factory'),
            s.inventory,
            distributed_delay(_orders(s), order=3, time=s.delivery_delay),
            label='deliveries',
        ),
        TransferLaw(s.inventory, Sink('customers'), _sales(s), label='sales'),
    ]


CarDealershipElement = Element.define(
    'CarDealershipElement',
    states={'inventory': state(default=200.0, domain=NumericDomain.NONNEGATIVE, quantity='cars')},
    parameters={
        'demand': parameter(default=20.0, domain=NumericDomain.NONNEGATIVE),
        'demand_step': parameter(default=2.0, domain=NumericDomain.REAL),
        'step_time': parameter(default=25.0, domain=NumericDomain.NONNEGATIVE),
        'coverage': parameter(default=10.0, domain=NumericDomain.POSITIVE),
        'perception_delay': parameter(default=5.0, domain=NumericDomain.POSITIVE),
        'response_delay': parameter(default=3.0, domain=NumericDomain.POSITIVE),
        'delivery_delay': parameter(default=5.0, domain=NumericDomain.POSITIVE),
    },
    laws=_dealer_laws,
)


@register_model(
    field='economics/markets',
    tags=('inventory', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: [
        observe_element('dealer', 'perceived', _perceived),
        observe_element('dealer', 'desired_inventory', _desired_inventory),
        observe_element('dealer', 'orders', _orders),
    ],
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
            The model, with the cars on the lot `dealer.inventory`, and the flows `dealer.deliveries` and
            `dealer.sales`. The observables of the model are `dealer.perceived` for the perceived sales,
            `dealer.desired_inventory`, and `dealer.orders`.

    Examples:
        >>> from skcomplex.models.economics.markets import car_dealer
        >>> car_dealer(response_delay=6.0).elements[0].response_delay
        6.0
    """
    dealer = CarDealershipElement(
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
