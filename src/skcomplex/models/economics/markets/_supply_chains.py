"""Markets models: Sterman's beer distribution game, a supply chain of four echelons with delays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import pairwise
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import TransferInteraction
from ....core.laws import BaseLaw, RateLaw, Sink, Source
from ....core.laws.utils import distributed_delay, maximum, minimum, step
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')


def _shipments(echelon: SimpleNamespace) -> Expression:
    """Return the shipments of an echelon, its backlog limited by its inventory over the dispatch time."""
    return minimum(echelon.backlog, echelon.inventory) / echelon.dispatch_time


def _orders(echelon: SimpleNamespace) -> Expression:
    """Return the orders of an echelon by Sterman's anchoring and adjustment rule, never negative."""
    effective = echelon.inventory - echelon.backlog
    orders = echelon.expected + echelon.alpha * (echelon.desired - effective - echelon.beta * echelon.supply_line)
    return maximum(orders, 0)


def _echelon_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the laws of an echelon, the shipping of its backlog and the decay of its expectation."""
    return [
        RateLaw(s.backlog, -_shipments(s)),
        RateLaw(s.expected, -s.theta * s.expected),
    ]


def _customer_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the customers, their demand after its step."""
    return s.demand + step(s.t, s.step_time, s.demand_step)


class EchelonElement(Element):
    """Echelon of a supply chain, which ships its backlog from its inventory and orders by anchoring and adjustment.

    It ships its backlog over the dispatch time, limited by its inventory. Its expected demand adapts to incoming
    orders at the rate `theta`, and it orders its expected demand plus `alpha` times the gap between its desired stock
    and its effective inventory, the inventory less the backlog, less `beta` times its supply line, never negative.

    Args:
        label:
            The label of the echelon.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        inventory:
            The initial inventory, in cases.

        in_transit:
            The initial beer on its way to the echelon.

        backlog:
            The initial orders waiting to ship.

        expected:
            The initial expected demand.

        supply_line:
            The initial beer ordered and not received.

        desired:
            The desired stock `S'`.

        theta:
            The rate at which the expected demand adapts to incoming orders.

        alpha:
            The fraction of the stock gap ordered per week.

        beta:
            The weight of the supply line in the stock gap.

        dispatch_time:
            The time over which the echelon ships its backlog.

    Examples:
        >>> from skcomplex.models.economics.markets import EchelonElement
        >>> EchelonElement('retailer', inventory=10.0).inventory
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        inventory: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='beer')] = 12.0,
        in_transit: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='beer')] = 8.0,
        backlog: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        expected: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        supply_line: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 16.0,
        desired: Annotated[float, ParameterSpec()] = 13.44,
        theta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.36,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.26,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.34,
        dispatch_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
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
        """Return the shipping of the backlog and the decay of the expected demand.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return _echelon_laws(s)


class SupplyLinkInteraction(TransferInteraction):
    """Link of a supply chain, which mails the orders of the downstream echelon upstream and ships beer back.

    Orders reach the upstream echelon through a third-order delay of the mail delay, raising its backlog and its
    expected demand, and its shipments reach the downstream echelon through a third-order delay of the shipping delay.
    The shipments and their arrivals are the legs of one transfer, from the inventory of the upstream echelon to the
    beer in transit to the downstream echelon and on to its inventory.

    Args:
        label:
            The label of the link.

        downstream:
            The path of the echelon that orders.

        upstream:
            The path of the echelon that ships.

        mail_delay:
            The average time an order takes to reach the upstream echelon.

        shipping_delay:
            The average time a shipment takes to reach the downstream echelon.

    Examples:
        >>> from skcomplex.models.economics.markets import SupplyLinkInteraction
        >>> SupplyLinkInteraction('link', downstream='retailer', upstream='wholesaler').mail_delay
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        downstream: Annotated[str, RoleSpec()],
        upstream: Annotated[str, RoleSpec()],
        mail_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        shipping_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(
            label,
            source=upstream,
            target=downstream,
            state='inventory',
            into='in_transit',
            amount=self._get_shipments,
            flow='shipments',
            legs=[('downstream.in_transit', 'downstream.inventory', self._get_arrivals, 'arrivals')],
        )
        self.downstream = downstream
        self.upstream = upstream
        self.mail_delay = mail_delay
        self.shipping_delay = shipping_delay

    @staticmethod
    def _get_shipments(s: SimpleNamespace) -> Expression:
        """Return the shipments of the upstream echelon."""
        return _shipments(s.upstream)

    @staticmethod
    def _get_arrivals(s: SimpleNamespace) -> Expression:
        """Return the arrivals at the downstream echelon, the shipments after a third-order shipping delay."""
        return distributed_delay(_shipments(s.upstream), order=3, time=s.shipping_delay)

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the upstream echelon, which ships, and the downstream echelon, which receives."""
        return [(namespace.upstream, namespace.downstream)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the mail of orders upstream and the shipping of beer downstream.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        orders = _orders(namespace.downstream)
        received = distributed_delay(orders, order=3, time=namespace.mail_delay)
        upstream = namespace.upstream
        return [
            RateLaw(namespace.downstream.supply_line, orders - self._get_arrivals(namespace)),
            RateLaw(upstream.backlog, received),
            RateLaw(upstream.expected, upstream.theta * received),
            *super()._get_laws(namespace),
        ]


class RetailDemandInteraction(TransferInteraction):
    """Customers of a supply chain, whose demand reaches the retailer and who take its shipments.

    The sales are a transfer from the inventory of the retailer to the customers.

    Args:
        label:
            The label of the customers.

        retailer:
            The path of the retailer.

        demand:
            The customer demand, in cases a week.

        demand_step:
            The permanent change of the demand.

        step_time:
            The week of the change of the demand.

    Examples:
        >>> from skcomplex.models.economics.markets import RetailDemandInteraction
        >>> RetailDemandInteraction('customers', retailer='retailer').demand
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        retailer: Annotated[str, RoleSpec()],
        demand: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        demand_step: Annotated[float, ParameterSpec()] = 4.0,
        step_time: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(
            label,
            source=retailer,
            target=retailer,
            state='inventory',
            into=Sink('customers'),
            amount=self._get_sales,
            flow='sales',
        )
        self.retailer = retailer
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time

    @staticmethod
    def _get_sales(s: SimpleNamespace) -> Expression:
        """Return the sales, the shipments of the retailer."""
        return _shipments(s.retailer)

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the retailer, whose inventory the customers take."""
        return [(namespace.retailer, namespace.retailer)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the demand that reaches the retailer and the sales to the customers.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        demand = _customer_orders(namespace)
        retailer = namespace.retailer
        return [
            RateLaw(retailer.backlog, demand),
            RateLaw(retailer.expected, retailer.theta * demand),
            *super()._get_laws(namespace),
        ]


class BreweryInteraction(TransferInteraction):
    """Brewery of a supply chain, which brews the orders of the factory over the production delay.

    The brewing is a transfer from the brewery into the beer in production at the factory and on to its inventory.

    Args:
        label:
            The label of the brewery.

        factory:
            The path of the factory.

        production_delay:
            The average time the brewery takes to brew an order, through a third-order delay.

    Examples:
        >>> from skcomplex.models.economics.markets import BreweryInteraction
        >>> BreweryInteraction('brewery', factory='factory').production_delay
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        factory: Annotated[str, RoleSpec()],
        production_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(
            label,
            source=factory,
            target=factory,
            state=Source('brewery'),
            into='in_transit',
            amount=self._get_starts,
            flow='starts',
            legs=[('in_transit', 'inventory', self._get_completions, 'completions')],
        )
        self.factory = factory
        self.production_delay = production_delay

    @staticmethod
    def _get_starts(s: SimpleNamespace) -> Expression:
        """Return the brews started, the orders of the factory."""
        return _orders(s.factory)

    @staticmethod
    def _get_completions(s: SimpleNamespace) -> Expression:
        """Return the brews completed, the orders after a third-order production delay."""
        return distributed_delay(_orders(s.factory), order=3, time=s.production_delay)

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the factory, whose beer in production and inventory the brewery fills."""
        return [(namespace.factory, namespace.factory)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the brewing of the orders of the factory.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        orders = _orders(namespace.factory)
        return [
            RateLaw(namespace.factory.supply_line, orders - self._get_completions(namespace)),
            *super()._get_laws(namespace),
        ]


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
