"""Markets models: Sterman's beer distribution game, a supply chain of four echelons with delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import pairwise

from ....core.base import Domain, Expression, parameter, state
from ....core.model import (
    Auxiliary,
    Element,
    Interaction,
    Law,
    Rate,
    Sink,
    Source,
    Symbols,
    System,
    Transfer,
    distributed_delay,
    step,
    where,
)
from ....core.runtime import register_model

ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')


def _shipments(echelon: Symbols) -> Expression:
    """Return the shipments of an echelon, its backlog limited by its inventory over the dispatch time."""
    available = where(echelon.backlog < echelon.inventory, echelon.backlog, echelon.inventory)
    return available / echelon.dispatch_time


def _orders(echelon: Symbols) -> Expression:
    """Return the orders of an echelon by Sterman's anchoring and adjustment rule, never negative."""
    effective = echelon.inventory - echelon.backlog
    orders = echelon.expected + echelon.alpha * (echelon.desired - effective - echelon.beta * echelon.supply_line)
    return where(orders > 0, orders, 0)


def _echelon_laws(s: Symbols) -> list[Law]:
    """Return the laws of an echelon, its orders and shipments and the decay of its expectation."""
    shipments = Auxiliary('shipments', _shipments(s))
    return [
        Auxiliary('orders', _orders(s)),
        shipments,
        Rate(s.backlog, -shipments.symbol),
        Rate(s.expected, -s.theta * s.expected),
    ]


def _link_laws(s: Symbols) -> list[Law]:
    """Return the laws of a link, the mail of orders upstream and the shipping of beer downstream."""
    orders = _orders(s.downstream)
    shipments = _shipments(s.upstream)
    received = distributed_delay(orders, order=3, time=s.mail_delay)
    arrivals = distributed_delay(shipments, order=3, time=s.shipping_delay)
    return [
        Rate(s.downstream.supply_line, orders - arrivals),
        Rate(s.upstream.backlog, received),
        Rate(s.upstream.expected, s.upstream.theta * received),
        Transfer(s.upstream.inventory, s.downstream.in_transit, shipments, label='shipments'),
        Transfer(s.downstream.in_transit, s.downstream.inventory, arrivals, label='arrivals'),
    ]


def _customer_laws(s: Symbols) -> list[Law]:
    """Return the laws of the customers, whose demand reaches the retailer and who take its shipments."""
    demand = Auxiliary('demand', s.demand + step(s.t, s.step_time, s.demand_step))
    return [
        demand,
        Rate(s.retailer.backlog, demand.symbol),
        Rate(s.retailer.expected, s.retailer.theta * demand.symbol),
        Transfer(s.retailer.inventory, Sink('customers'), _shipments(s.retailer), label='sales'),
    ]


def _brewery_laws(s: Symbols) -> list[Law]:
    """Return the laws of the brewery, which brews the orders of the factory over the production delay."""
    orders = _orders(s.factory)
    completions = distributed_delay(orders, order=3, time=s.production_delay)
    return [
        Rate(s.factory.supply_line, orders - completions),
        Transfer(Source('brewery'), s.factory.in_transit, orders, label='starts'),
        Transfer(s.factory.in_transit, s.factory.inventory, completions, label='completions'),
    ]


Echelon = Element.define(
    'Echelon',
    states={
        'inventory': state(default=12.0, domain=Domain.NONNEGATIVE, quantity='beer'),
        'in_transit': state(default=8.0, domain=Domain.NONNEGATIVE, quantity='beer'),
        'backlog': state(default=4.0, domain=Domain.NONNEGATIVE),
        'expected': state(default=4.0, domain=Domain.NONNEGATIVE),
        'supply_line': state(default=16.0, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'desired': parameter(default=13.44, domain=Domain.REAL),
        'theta': parameter(default=0.36, domain=Domain.NONNEGATIVE),
        'alpha': parameter(default=0.26, domain=Domain.NONNEGATIVE),
        'beta': parameter(default=0.34, domain=Domain.NONNEGATIVE),
        'dispatch_time': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=_echelon_laws,
)

SupplyLink = Interaction.define(
    'SupplyLink',
    roles=['downstream', 'upstream'],
    parameters={
        'mail_delay': parameter(default=2.0, domain=Domain.POSITIVE),
        'shipping_delay': parameter(default=2.0, domain=Domain.POSITIVE),
    },
    laws=_link_laws,
)

RetailDemand = Interaction.define(
    'RetailDemand',
    roles=['retailer'],
    parameters={
        'demand': parameter(default=4.0, domain=Domain.NONNEGATIVE),
        'demand_step': parameter(default=4.0, domain=Domain.REAL),
        'step_time': parameter(default=5.0, domain=Domain.NONNEGATIVE),
    },
    laws=_customer_laws,
)

Brewery = Interaction.define(
    'Brewery',
    roles=['factory'],
    parameters={'production_delay': parameter(default=3.0, domain=Domain.POSITIVE)},
    laws=_brewery_laws,
)


@register_model(
    field='economics/markets',
    tags=('supply chain', 'bullwhip effect', 'delay', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Sterman 1989',),
    defaults={'until': 36.0, 'dt': 0.5},
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
            `<echelon>.expected`, and `<echelon>.supply_line` and the auxiliaries `<echelon>.orders` and
            `<echelon>.shipments` of the echelons `retailer`, `wholesaler`, `distributor`, and `factory`, the flows
            `<upstream>_to_<downstream>.shipments` and `<upstream>_to_<downstream>.arrivals` of the links,
            `customers.sales` and `customers.demand`, and `brewery.starts` and `brewery.completions`.

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
        echelon = Echelon(
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
        SupplyLink(
            f'{upstream}_to_{downstream}',
            downstream=downstream,
            upstream=upstream,
            mail_delay=mail_delay,
            shipping_delay=shipping_delay,
        )
        for downstream, upstream in pairwise(ECHELONS)
    ]
    customers = RetailDemand(
        'customers',
        retailer='retailer',
        demand=demand,
        demand_step=demand_step,
        step_time=step_time,
    )
    brewery = Brewery('brewery', factory='factory', production_delay=production_delay)
    return System(components=echelons, interactions=[*links, customers, brewery])
