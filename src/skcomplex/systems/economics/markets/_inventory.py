"""Markets systems: a retailer's inventory under delays.

The delays are of perception, response, and delivery.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.laws.utils import distributed_delay, maximum, minimum, step
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


def _sales(s: SimpleNamespace) -> Expression:
    """Return the sales of the car dealership, the demand up to the inventory."""
    demand = s.demand + step(s.t, s.step_time, s.demand_step)
    return minimum(demand, s.inventory)


def _perceived(s: SimpleNamespace) -> Expression:
    """Return the perceived sales of the car dealership.

    They are its sales smoothed over the perception delay.
    """
    return distributed_delay(_sales(s), conserved=False, time=s.perception_delay)


def _desired_inventory(s: SimpleNamespace) -> Expression:
    """Return the inventory the car dealership desires.

    It is a coverage of its perceived sales.
    """
    return s.coverage * _perceived(s)


def _orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the car dealership.

    They are its perceived sales and the gap to its desired inventory.
    """
    gap_orders = _perceived(s) + (_desired_inventory(s) - s.inventory) / s.response_delay
    return maximum(gap_orders, 0)


def _dealer(s: SimpleNamespace) -> SimpleNamespace:
    """Return the symbols of the dealership a transfer moves cars of, with the time."""
    return SimpleNamespace(t=s.t, **vars(s.source))


class CarDealershipElement(Element):
    """Car dealership whose inventory deliveries fill and sales drain.

    Its orders close the gap to a desired coverage. Sales are the demand, stepping up at `step_time`, limited by the
    inventory. The dealer perceives sales through a first-order smoothing, desires `coverage` days of perceived sales,
    and orders the perceived sales plus the gap to the desired inventory over the response delay, never negative. Orders
    arrive through a third-order delay. The dealership holds the inventory and the parameters, and a transfer within it
    of the system moves the cars.

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
        >>> from skcomplex.systems.economics.markets import CarDealershipElement
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


class CarDealerSystem(BaseSystem):
    """Meadows' car dealership.

    Its inventory oscillates after a step in demand because of its delays. The inventory is a stock of cars that
    deliveries from the factory fill and sales to customers drain. Sales are the customer demand, limited by the cars on
    the lot. The dealer smooths sales over the perception delay, wants an inventory of `coverage` days of perceived
    sales, and orders the perceived sales plus the gap to that inventory divided by the response delay. Orders arrive
    through a third-order material delay, a continuous stand-in for Meadows' fixed delivery delay. Lengthening the
    response delay damps the oscillation, while shortening the perception delay makes the dealer react faster to the new
    demand and slightly enlarges it.

    The system has the cars on the lot `dealer.inventory`, and the flows `dealer.deliveries` and `dealer.sales`. The
    observables of the system are `dealer.perceived` for the perceived sales, `dealer.desired_inventory`, and
    `dealer.orders`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import CarDealerSystem
        >>> CarDealerSystem(response_delay=6.0).prepare().system_.elements[0].response_delay
        6.0
    """

    field = 'economics/markets'
    topics = ('inventory', 'delay', 'oscillations', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Meadows 2008',)
    until = 100.0
    dt = 1.0
    observables = (
        ObservableSpec('perceived', _perceived, element='dealer'),
        ObservableSpec('desired_inventory', _desired_inventory, element='dealer'),
        ObservableSpec('orders', _orders, element='dealer'),
    )

    def __init__(
        self,
        inventory: float = 200.0,
        *,
        demand: float = 20.0,
        demand_step: float = 2.0,
        step_time: float = 25.0,
        coverage: float = 10.0,
        perception_delay: float = 5.0,
        response_delay: float = 3.0,
        delivery_delay: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.inventory = inventory
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time
        self.coverage = coverage
        self.perception_delay = perception_delay
        self.response_delay = response_delay
        self.delivery_delay = delivery_delay

    @staticmethod
    def _deliveries(s: SimpleNamespace) -> Expression:
        """Return the deliveries of the dealership, its orders through a third-order delay."""
        dealer = _dealer(s)
        return distributed_delay(_orders(dealer), order=3, time=dealer.delivery_delay)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        dealer = CarDealershipElement(
            'dealer',
            inventory=self.inventory,
            demand=self.demand,
            demand_step=self.demand_step,
            step_time=self.step_time,
            coverage=self.coverage,
            perception_delay=self.perception_delay,
            response_delay=self.response_delay,
            delivery_delay=self.delivery_delay,
        )
        cars = TransferInteraction(
            'cars',
            source='dealer',
            target='dealer',
            state=Source('factory'),
            into='inventory',
            amount=self._deliveries,
            flow='deliveries',
            legs=[('inventory', Sink('customers'), lambda s: _sales(_dealer(s)), 'sales')],
        )
        return System(self.label, elements=[dealer], interactions=[cars])
