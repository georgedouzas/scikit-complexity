"""Markets models: Sterman's beer distribution game, a supply chain of four echelons with delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import pairwise
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.laws.utils import distributed_delay, step, where
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')


def _shipments(echelon: SimpleNamespace) -> Expression:
    """Return the shipments of an echelon, its backlog limited by its inventory over the dispatch time."""
    available = where(echelon.backlog < echelon.inventory, echelon.backlog, echelon.inventory)
    return available / echelon.dispatch_time


def _orders(echelon: SimpleNamespace) -> Expression:
    """Return the orders of an echelon by Sterman's anchoring and adjustment rule, never negative."""
    effective = echelon.inventory - echelon.backlog
    orders = echelon.expected + echelon.alpha * (echelon.desired - effective - echelon.beta * echelon.supply_line)
    return where(orders > 0, orders, 0)


def _echelon_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of an echelon, the shipping of its backlog and the decay of its expectation."""
    return [
        RateLaw(s.backlog, -_shipments(s)),
        RateLaw(s.expected, -s.theta * s.expected),
    ]


def _link_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of a link, the mail of orders upstream and the shipping of beer downstream."""
    orders = _orders(s.downstream)
    shipments = _shipments(s.upstream)
    received = distributed_delay(orders, order=3, time=s.mail_delay)
    arrivals = distributed_delay(shipments, order=3, time=s.shipping_delay)
    return [
        RateLaw(s.downstream.supply_line, orders - arrivals),
        RateLaw(s.upstream.backlog, received),
        RateLaw(s.upstream.expected, s.upstream.theta * received),
        TransferLaw(s.upstream.inventory, s.downstream.in_transit, shipments, label='shipments'),
        TransferLaw(s.downstream.in_transit, s.downstream.inventory, arrivals, label='arrivals'),
    ]


def _customer_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the customers, their demand after its step."""
    return s.demand + step(s.t, s.step_time, s.demand_step)


def _customer_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of the customers, whose demand reaches the retailer and who take its shipments."""
    demand = _customer_orders(s)
    return [
        RateLaw(s.retailer.backlog, demand),
        RateLaw(s.retailer.expected, s.retailer.theta * demand),
        TransferLaw(s.retailer.inventory, Sink('customers'), _shipments(s.retailer), label='sales'),
    ]


def _brewery_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of the brewery, which brews the orders of the factory over the production delay."""
    orders = _orders(s.factory)
    completions = distributed_delay(orders, order=3, time=s.production_delay)
    return [
        RateLaw(s.factory.supply_line, orders - completions),
        TransferLaw(Source('brewery'), s.factory.in_transit, orders, label='starts'),
        TransferLaw(s.factory.in_transit, s.factory.inventory, completions, label='completions'),
    ]


class EchelonElement(Element):
    """Echelon element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        inventory: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='beer')] = 12.0,
        in_transit: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='beer')] = 8.0,
        backlog: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        expected: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        supply_line: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 16.0,
        desired: Annotated[object, ParameterSpec()] = 13.44,
        theta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.36,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.26,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.34,
        dispatch_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.inventory = inventory
        self.in_transit = in_transit
        self.backlog = backlog
        self.expected = expected
        self.supply_line = supply_line
        self.desired = desired
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.dispatch_time = dispatch_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the echelon element."""
        return _echelon_laws(s)


class SupplyLinkInteraction(Interaction):
    """Supply link interaction."""

    def __init__(
        self,
        label: str,
        *,
        downstream: Annotated[object, RoleSpec()],
        upstream: Annotated[object, RoleSpec()],
        mail_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        shipping_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.downstream = downstream
        self.upstream = upstream
        self.mail_delay = mail_delay
        self.shipping_delay = shipping_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the supply link interaction."""
        return _link_laws(s)


class RetailDemandInteraction(Interaction):
    """Retail demand interaction."""

    def __init__(
        self,
        label: str,
        *,
        retailer: Annotated[object, RoleSpec()],
        demand: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        demand_step: Annotated[object, ParameterSpec()] = 4.0,
        step_time: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label)
        self.retailer = retailer
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the retail demand interaction."""
        return _customer_laws(s)


class BreweryInteraction(Interaction):
    """Brewery interaction."""

    def __init__(
        self,
        label: str,
        *,
        factory: Annotated[object, RoleSpec()],
        production_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label)
        self.factory = factory
        self.production_delay = production_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the brewery interaction."""
        return _brewery_laws(s)


@register_model(
    field='economics/markets',
    tags=('supply chain', 'bullwhip effect', 'delay', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Sterman 1989',),
    defaults={'until': 36.0, 'dt': 0.5},
    observables=lambda system: [
        *(ObservableSpec('orders', _orders, element=label) for label in ECHELONS),
        *(ObservableSpec('shipments', _shipments, element=label) for label in ECHELONS),
        ObservableSpec('orders', _customer_orders, element='customers'),
    ],
)
def beer_game(
    inventory: float = 12.0,
    *,
    demand: float = 4.0,
    demand_step: float = 4.0,
    step_time: float = 5.0,
    theta: float = 0.36,
    alpha: float = 0.26,
    beta: float = 0.34,
    dispatch_time: float = 1.0,
    mail_delay: float = 2.0,
    shipping_delay: float = 2.0,
    production_delay: float = 3.0,
) -> System:
    """Return Sterman's beer distribution game, a supply chain whose orders swing wider at every echelon upstream.

    A retailer, a wholesaler, a distributor, and a factory each hold an inventory of beer and a backlog of orders
    waiting to ship. Each ships its backlog over the dispatch time, limited by its inventory, and the beer reaches the
    next echelon downstream after the shipping delay. Orders reach the next echelon upstream after the mail delay, and
    the factory brews its orders over the production delay. Customer demand steps from 4 to 8 cases a week in week 5.
    Every echelon orders by Sterman's anchoring and adjustment rule, its expected demand plus `alpha` times the gap
    between its desired stock `S'` and its effective inventory, the inventory less the backlog, plus `beta` times its
    supply line, the beer it ordered and has not received. The expected demand adapts to incoming orders at the rate
    `theta`. The defaults of `theta`, `alpha`, and `beta` are the median estimates of Sterman 1989, and `S'` is set so
    the chain starts in equilibrium at 4 cases a week. The game is weekly and its delays are fixed, while this model is
    a continuous approximation in which third-order delays stand in for the fixed ones. The amplitude of the orders
    grows from the retailer to the factory, the bullwhip effect.

    Args:
        inventory:
            The initial inventory of every echelon, in cases.

        demand:
            The customer demand, in cases a week.

        demand_step:
            The permanent change of the customer demand.

        step_time:
            The week of the change of the customer demand.

        theta:
            The rate at which the expected demand adapts to incoming orders.

        alpha:
            The fraction of the stock gap ordered per week.

        beta:
            The weight of the supply line in the stock gap.

        dispatch_time:
            The time over which an echelon ships its backlog.

        mail_delay:
            The average time an order takes to reach the next echelon upstream.

        shipping_delay:
            The average time a shipment takes to reach the next echelon downstream.

        production_delay:
            The average time the factory takes to brew its orders.

    Returns:
        system:
            The model, with the stocks `<echelon>.inventory`, `<echelon>.in_transit`, `<echelon>.backlog`,
            `<echelon>.expected`, and `<echelon>.supply_line` of the echelons `retailer`, `wholesaler`, `distributor`,
            and `factory`, and the flows `<upstream>_to_<downstream>.shipments` and
            `<upstream>_to_<downstream>.arrivals` of the links, `customers.sales`, and `brewery.starts` and
            `brewery.completions`. The observables of the model are the orders `<echelon>.orders` and the shipments
            `<echelon>.shipments` of every echelon, and the orders of the customers `customers.orders`.

    Examples:
        >>> from skcomplex.models.economics.markets import beer_game
        >>> [element.label for element in beer_game().elements]
        ['retailer', 'wholesaler', 'distributor', 'factory']
    """
    throughput = demand * dispatch_time
    echelons = []
    for label in ECHELONS:
        pipeline = production_delay if label == 'factory' else shipping_delay
        lead_time = pipeline if label == 'factory' else mail_delay + shipping_delay
        supply_line = demand * lead_time
        echelon = EchelonElement(
            label,
            inventory=inventory,
            in_transit=demand * pipeline,
            backlog=throughput,
            expected=demand,
            supply_line=supply_line,
            desired=inventory - throughput + beta * supply_line,
            theta=theta,
            alpha=alpha,
            beta=beta,
            dispatch_time=dispatch_time,
        )
        echelons.append(echelon)
    links = [
        SupplyLinkInteraction(
            f'{upstream}_to_{downstream}',
            downstream=downstream,
            upstream=upstream,
            mail_delay=mail_delay,
            shipping_delay=shipping_delay,
        )
        for downstream, upstream in pairwise(ECHELONS)
    ]
    customers = RetailDemandInteraction(
        'customers',
        retailer='retailer',
        demand=demand,
        demand_step=demand_step,
        step_time=step_time,
    )
    brewery = BreweryInteraction('brewery', factory='factory', production_delay=production_delay)
    return System(elements=echelons, interactions=[*links, customers, brewery])
