"""Markets models: a retailer's inventory under perception, response, and delivery delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.laws.utils import distributed_delay, maximum, minimum, step
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _sales(s: SimpleNamespace) -> Expression:
    """Return the sales of the car dealership, the demand up to the inventory."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    return minimum(demand, s.inventory)


def _perceived(s: SimpleNamespace) -> Expression:
    """Return the perceived sales of the car dealership, its sales smoothed over the perception delay."""
    return distributed_delay(_sales(s), conserved=False, time=s.perception_delay)


def _desired_inventory(s: SimpleNamespace) -> Expression:
    """Return the inventory the car dealership desires, a coverage of its perceived sales."""
    return s.coverage * _perceived(s)


def _orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the car dealership, its perceived sales and the gap to its desired inventory."""
    gap_orders = _perceived(s) + (_desired_inventory(s) - s.inventory) / s.response_delay
    return maximum(gap_orders, 0)


def _dealer_laws(s: SimpleNamespace) -> list[BaseLaw]:
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


class CarDealershipElement(Element):
    """Car dealership whose inventory deliveries fill and sales drain, ordered to close the gap to a desired coverage.

    Sales are the demand, stepping up at `step_time`, limited by the inventory. The dealer perceives sales through a
    first-order smoothing, desires `coverage` days of perceived sales, and orders the perceived sales plus the gap to
    the desired inventory over the response delay, never negative. Orders arrive through a third-order delay.

    Args:
        label:
            The label of the dealership.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.models.economics.markets import CarDealershipElement
        >>> CarDealershipElement('dealer', inventory=150.0).inventory
        150.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        inventory: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='cars')] = 200.0,
        demand: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        demand_step: Annotated[float, ParameterSpec()] = 2.0,
        step_time: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 25.0,
        coverage: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        perception_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        response_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        delivery_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.inventory = inventory
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time
        self.coverage = coverage
        self.perception_delay = perception_delay
        self.response_delay = response_delay
        self.delivery_delay = delivery_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the deliveries and the sales of the dealership.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _dealer_laws(s)


@register_model(
    field='economics/markets',
    tags=('inventory', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: [
        ObservableSpec('perceived', _perceived, element='dealer'),
        ObservableSpec('desired_inventory', _desired_inventory, element='dealer'),
        ObservableSpec('orders', _orders, element='dealer'),
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
    return System(elements=[dealer])
