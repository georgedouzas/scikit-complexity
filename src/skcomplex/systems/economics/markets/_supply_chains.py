"""Markets systems: Sterman's beer distribution game, a supply chain of four echelons with delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import pairwise
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import distributed_delay, step
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._brewery import BreweryInteraction
from ._echelon import EchelonElement
from ._factory import FactoryElement
from ._retail_demand import RetailDemandInteraction
from ._retailer import RetailerElement
from ._supply_link import SupplyLinkInteraction

ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')


def _received(downstream: SimpleNamespace) -> Expression:
    """Return the orders of the downstream echelon that reach its supplier, after a third-order mail delay."""
    return distributed_delay(BreweryInteraction._orders(downstream), order=3, time=downstream.mail_delay)


def _demand(retailer: SimpleNamespace, t: Expression) -> Expression:
    """Return the orders of the customers of the retailer, their demand after its step."""
    return retailer.demand + step(t, retailer.step_time, retailer.demand_step)


def _customer_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the customers of the namespace of the system, which holds the retailer."""
    return _demand(s.retailer, s.t)


def _get_link_rates(downstream: str, upstream: str) -> list[RateInteraction]:
    """Return the rates of a link, the supply line of the downstream echelon and the incoming orders upstream."""
    return [
        RateInteraction(
            f'{downstream}_supply_line',
            target=downstream,
            state='supply_line',
            sources=[upstream],
            rate=lambda s: BreweryInteraction._orders(s.target)
            - SupplyLinkInteraction._arrivals(s.sources[0], s.target),
        ),
        RateInteraction(
            f'{upstream}_backlog',
            target=upstream,
            state='backlog',
            sources=[downstream],
            rate=lambda s: _received(s.sources[0]),
        ),
        RateInteraction(
            f'{upstream}_expected',
            target=upstream,
            state='expected',
            sources=[downstream],
            rate=lambda s: s.target.theta * _received(s.sources[0]),
        ),
    ]


def _get_customer_rates() -> list[RateInteraction]:
    """Return the rates of the customers, the demand that reaches the retailer and its expectation."""
    return [
        RateInteraction('retailer_backlog', target='retailer', state='backlog', rate=lambda s: _demand(s.target, s.t)),
        RateInteraction(
            'retailer_expected',
            target='retailer',
            state='expected',
            rate=lambda s: s.target.theta * _demand(s.target, s.t),
        ),
    ]


def _get_brewery_rate() -> RateInteraction:
    """Return the rate of the supply line of the factory, its orders less the brews completed."""
    return RateInteraction(
        'factory_supply_line',
        target='factory',
        state='supply_line',
        rate=lambda s: BreweryInteraction._orders(s.target) - BreweryInteraction._completions(s.target),
    )


@register_system(
    field='economics/markets',
    tags=('supply chain', 'bullwhip effect', 'delay', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Sterman 1989',),
    defaults={'until': 36.0, 'dt': 0.5},
    observables=lambda system: [
        *(ObservableSpec('orders', BreweryInteraction._orders, element=label) for label in ECHELONS),
        *(ObservableSpec('shipments', EchelonElement._shipments, element=label) for label in ECHELONS),
        ObservableSpec('customers.orders', _customer_orders),
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
    the chain starts in equilibrium at 4 cases a week. The game is weekly and its delays are fixed, while this system is
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
            The system, with the stocks `<echelon>.inventory`, `<echelon>.in_transit`, `<echelon>.backlog`,
            `<echelon>.expected`, and `<echelon>.supply_line` of the echelons `retailer`, `wholesaler`, `distributor`,
            and `factory`, and the flows `<upstream>_to_<downstream>.shipments` and
            `<upstream>_to_<downstream>.arrivals` of the links, `customers.sales`, and `brewery.starts` and
            `brewery.completions`. The rates `<echelon>_backlog` and `<echelon>_expected` add the orders that reach
            every echelon and adapt its expectation, and the rates `<echelon>_supply_line` keep its supply line. The
            observables of the system are the orders `<echelon>.orders` and the shipments
            `<echelon>.shipments` of every echelon, and the orders of the customers `customers.orders`.

    Examples:
        >>> from skcomplex.systems.economics.markets import beer_game
        >>> [element.label for element in beer_game().elements]
        ['retailer', 'wholesaler', 'distributor', 'factory']
    """
    throughput = demand * dispatch_time
    echelons: list[EchelonElement] = []
    for label in ECHELONS:
        pipeline = production_delay if label == 'factory' else shipping_delay
        lead_time = pipeline if label == 'factory' else mail_delay + shipping_delay
        supply_line = demand * lead_time
        values = {
            'inventory': inventory,
            'in_transit': demand * pipeline,
            'backlog': throughput,
            'expected': demand,
            'supply_line': supply_line,
            'desired': inventory - throughput + beta * supply_line,
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'dispatch_time': dispatch_time,
            'mail_delay': mail_delay,
            'shipping_delay': shipping_delay,
        }
        if label == 'retailer':
            values.update(demand=demand, demand_step=demand_step, step_time=step_time)
        elif label == 'factory':
            values.update(production_delay=production_delay)
        kind = {'retailer': RetailerElement, 'factory': FactoryElement}.get(label, EchelonElement)
        echelons.append(kind(label).set_params(**values))
    links = [
        interaction
        for downstream, upstream in pairwise(ECHELONS)
        for interaction in (
            *_get_link_rates(downstream, upstream),
            SupplyLinkInteraction(f'{upstream}_to_{downstream}', downstream=downstream, upstream=upstream),
        )
    ]
    customers = [*_get_customer_rates(), RetailDemandInteraction('customers', retailer='retailer')]
    brewery = [_get_brewery_rate(), BreweryInteraction('brewery', factory='factory')]
    return System(elements=echelons, interactions=[*links, *customers, *brewery])
