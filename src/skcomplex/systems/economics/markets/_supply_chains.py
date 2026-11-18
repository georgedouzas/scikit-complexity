"""Markets systems: Sterman's beer distribution game.

It is a supply chain of four echelons with delays.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from itertools import pairwise
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Quantity
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import distributed_delay, maximum, minimum, step
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System

ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')


class EchelonElement(Element):
    """Echelon of a supply chain.

    It ships its backlog from its inventory and orders by anchoring and adjustment, by the laws of `EchelonInteraction`.
    It ships its backlog over the dispatch time, limited by its inventory. Its expected demand adapts to incoming orders
    at the rate `theta`, and it orders its expected demand plus `alpha` times the gap between its desired stock and its
    effective inventory, the inventory less the backlog, less `beta` times its supply line, never negative. Its orders
    reach its supplier after the mail delay, and the beer its supplier ships arrives after the shipping delay.

    Args:
        label:
            The label of the echelon.

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

        mail_delay:
            The average time an order of the echelon takes to reach its supplier.

        shipping_delay:
            The average time a shipment takes to reach the echelon from its supplier.

    Examples:
        >>> from skcomplex.systems.economics.markets import EchelonElement
        >>> EchelonElement('retailer', inventory=10.0).inventory
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        inventory: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 12.0,
        in_transit: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 8.0,
        backlog: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        expected: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        supply_line: Annotated[float, RealSet.closed_open(0, oo)] = 16.0,
        desired: float = 13.44,
        theta: Annotated[float, RealSet.closed_open(0, oo)] = 0.36,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 0.26,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.34,
        dispatch_time: Annotated[float, RealSet.open(0, oo)] = 1.0,
        mail_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
        shipping_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
    ) -> None:
        super().__init__(label)
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
        self.mail_delay = mail_delay
        self.shipping_delay = shipping_delay

    @staticmethod
    def _get_shipments(echelon: SimpleNamespace) -> Expression:
        """Return the shipments of an echelon.

        They are its backlog limited by its inventory over the dispatch time.
        """
        return minimum(echelon.backlog, echelon.inventory) / echelon.dispatch_time


class FactoryElement(EchelonElement):
    """Factory of a supply chain.

    It is an echelon whose orders a brewery brews over the production delay.

    Args:
        label:
            The label of the factory.

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

        mail_delay:
            The average time an order of the echelon takes to reach its supplier.

        shipping_delay:
            The average time a shipment takes to reach the echelon from its supplier.

        production_delay:
            The average time the brewery takes to brew an order, through a third-order delay.

    Examples:
        >>> from skcomplex.systems.economics.markets import FactoryElement
        >>> FactoryElement('factory').production_delay
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        inventory: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 12.0,
        in_transit: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 8.0,
        backlog: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        expected: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        supply_line: Annotated[float, RealSet.closed_open(0, oo)] = 16.0,
        desired: float = 13.44,
        theta: Annotated[float, RealSet.closed_open(0, oo)] = 0.36,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 0.26,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.34,
        dispatch_time: Annotated[float, RealSet.open(0, oo)] = 1.0,
        mail_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
        shipping_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
        production_delay: Annotated[float, RealSet.open(0, oo)] = 3.0,
    ) -> None:
        super().__init__(
            label,
            inventory=inventory,
            in_transit=in_transit,
            backlog=backlog,
            expected=expected,
            supply_line=supply_line,
            desired=desired,
            theta=theta,
            alpha=alpha,
            beta=beta,
            dispatch_time=dispatch_time,
            mail_delay=mail_delay,
            shipping_delay=shipping_delay,
        )
        self.production_delay = production_delay


class RetailerElement(EchelonElement):
    """Retailer of a supply chain.

    It is an echelon whose customers order their demand, which steps once.

    Args:
        label:
            The label of the retailer.

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

        mail_delay:
            The average time an order of the echelon takes to reach its supplier.

        shipping_delay:
            The average time a shipment takes to reach the echelon from its supplier.

        demand:
            The customer demand, in cases a week.

        demand_step:
            The permanent change of the demand.

        step_time:
            The week of the change of the demand.

    Examples:
        >>> from skcomplex.systems.economics.markets import RetailerElement
        >>> RetailerElement('retailer').demand
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        inventory: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 12.0,
        in_transit: Annotated[float, RealSet.closed_open(0, oo), Quantity('beer')] = 8.0,
        backlog: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        expected: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        supply_line: Annotated[float, RealSet.closed_open(0, oo)] = 16.0,
        desired: float = 13.44,
        theta: Annotated[float, RealSet.closed_open(0, oo)] = 0.36,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 0.26,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.34,
        dispatch_time: Annotated[float, RealSet.open(0, oo)] = 1.0,
        mail_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
        shipping_delay: Annotated[float, RealSet.open(0, oo)] = 2.0,
        demand: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        demand_step: float = 4.0,
        step_time: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
    ) -> None:
        super().__init__(
            label,
            inventory=inventory,
            in_transit=in_transit,
            backlog=backlog,
            expected=expected,
            supply_line=supply_line,
            desired=desired,
            theta=theta,
            alpha=alpha,
            beta=beta,
            dispatch_time=dispatch_time,
            mail_delay=mail_delay,
            shipping_delay=shipping_delay,
        )
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time


class EchelonInteraction(Interaction):
    """Shipping of the backlog of an echelon of a supply chain and the decay of its expected demand.

    The echelon ships its backlog over the dispatch time, limited by its inventory, and its expected demand decays at
    the rate `theta`, while the orders that reach it raise both.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the echelon.

    Examples:
        >>> from skcomplex.systems.economics.markets import EchelonInteraction
        >>> EchelonInteraction('retailer_shipping', target='retailer').target
        'retailer'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the shipping of the backlog and the decay of the expectation as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            echelon = context.target
            self.laws_: list[BaseComponent] = [
                RateLaw(echelon.backlog, -EchelonElement._get_shipments(echelon)),
                RateLaw(echelon.expected, -echelon.theta * echelon.expected),
            ]


class BreweryInteraction(Interaction):
    """Brewery of a supply chain.

    It brews the orders of the factory over its production delay. The brewing is a flow `starts` into the beer in
    production at the factory and a flow `completions` on to its inventory.

    Args:
        label:
            The label of the brewery.

        factory:
            The path of the factory.

    Examples:
        >>> from skcomplex.systems.economics.markets import BreweryInteraction
        >>> BreweryInteraction('brewery', factory='factory').factory
        'factory'
    """

    def __init__(
        self,
        label: str,
        *,
        factory: Role,
    ) -> None:
        super().__init__(label)
        self.factory = factory

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the starts and the completions of the brews as the flows."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            factory = context.factory
            self.laws_: list[BaseComponent] = [
                RateLaw({factory.in_transit: +1}, BreweryInteraction._get_orders(factory), label='starts'),
                RateLaw(
                    {factory.in_transit: -1, factory.inventory: +1},
                    BreweryInteraction._get_completions(factory),
                    label='completions',
                ),
            ]

    @staticmethod
    def _get_orders(echelon: SimpleNamespace) -> Expression:
        """Return the orders of an echelon by Sterman's anchoring and adjustment rule.

        They are never negative.
        """
        effective = echelon.inventory - echelon.backlog
        orders = echelon.expected + echelon.alpha * (echelon.desired - effective - echelon.beta * echelon.supply_line)
        return maximum(orders, 0)

    @staticmethod
    def _get_completions(factory: SimpleNamespace) -> Expression:
        """Return the brews completed for the factory.

        They are its orders after a third-order production delay.
        """
        return distributed_delay(BreweryInteraction._get_orders(factory), order=3, time=factory.production_delay)


class RetailDemandInteraction(Interaction):
    """Customers of a supply chain, who take the shipments of the retailer.

    The sales are a flow out of the inventory of the retailer.

    Args:
        label:
            The label of the customers.

        retailer:
            The path of the retailer.

    Examples:
        >>> from skcomplex.systems.economics.markets import RetailDemandInteraction
        >>> RetailDemandInteraction('customers', retailer='retailer').retailer
        'retailer'
    """

    def __init__(
        self,
        label: str,
        *,
        retailer: Role,
    ) -> None:
        super().__init__(label)
        self.retailer = retailer

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the sales out of the inventory of the retailer as the flow."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            retailer = context.retailer
            self.laws_: list[BaseComponent] = [
                RateLaw({retailer.inventory: -1}, EchelonElement._get_shipments(retailer), label='sales'),
            ]


class SupplyLinkInteraction(Interaction):
    """Link of a supply chain.

    It ships beer from the upstream echelon to the downstream echelon. The shipments flow from the inventory of the
    upstream echelon to the beer in transit to the downstream echelon, and the arrivals, after a third-order delay of
    the shipping delay of the downstream echelon, flow on to its inventory.

    Args:
        label:
            The label of the link.

        downstream:
            The path of the echelon that orders.

        upstream:
            The path of the echelon that ships.

    Examples:
        >>> from skcomplex.systems.economics.markets import SupplyLinkInteraction
        >>> SupplyLinkInteraction('link', downstream='retailer', upstream='wholesaler').upstream
        'wholesaler'
    """

    def __init__(
        self,
        label: str,
        *,
        downstream: Role,
        upstream: Role,
    ) -> None:
        super().__init__(label)
        self.downstream = downstream
        self.upstream = upstream

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the shipments and their arrivals as the flows."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            upstream, downstream = context.upstream, context.downstream
            self.laws_: list[BaseComponent] = [
                RateLaw(
                    {upstream.inventory: -1, downstream.in_transit: +1},
                    EchelonElement._get_shipments(upstream),
                    label='shipments',
                ),
                RateLaw(
                    {downstream.in_transit: -1, downstream.inventory: +1},
                    SupplyLinkInteraction._get_arrivals(upstream, downstream),
                    label='arrivals',
                ),
            ]

    @staticmethod
    def _get_arrivals(upstream: SimpleNamespace, downstream: SimpleNamespace) -> Expression:
        """Return the arrivals at the downstream echelon.

        They are the shipments after a third-order shipping delay.
        """
        return distributed_delay(EchelonElement._get_shipments(upstream), order=3, time=downstream.shipping_delay)


def _get_customer_orders(s: SimpleNamespace) -> Expression:
    """Return the orders of the customers of the namespace of the system.

    The namespace holds the retailer.
    """
    return _get_demand(s.retailer, s.t)


def _get_demand(retailer: SimpleNamespace, t: Expression) -> Expression:
    """Return the orders of the customers of the retailer, their demand after its step."""
    return retailer.demand + step(t, retailer.step_time, retailer.demand_step)


class BeerGameSystem(BaseSystem):
    """Sterman's beer distribution game.

    Its orders swing wider at every echelon upstream. A retailer, a wholesaler, a distributor, and a factory each hold
    an inventory of beer and a backlog of orders waiting to ship. Each ships its backlog over the dispatch time, limited
    by its inventory, and the beer reaches the next echelon downstream after the shipping delay. Orders reach the next
    echelon upstream after the mail delay, and the factory brews its orders over the production delay. Customer demand
    steps from 4 to 8 cases a week in week 5. Every echelon orders by Sterman's anchoring and adjustment rule, its
    expected demand plus `alpha` times the gap between its desired stock `S'` and its effective inventory, the inventory
    less the backlog, plus `beta` times its supply line, the beer it ordered and has not received. The expected demand
    adapts to incoming orders at the rate `theta`. The defaults of `theta`, `alpha`, and `beta` are the median estimates
    of Sterman 1989, and `S'` is set so the chain starts in equilibrium at 4 cases a week. The game is weekly and its
    delays are fixed, while this system is a continuous approximation in which third-order delays stand in for the fixed
    ones. The amplitude of the orders grows from the retailer to the factory, the bullwhip effect.

    The system has the stocks `<echelon>.inventory`, `<echelon>.in_transit`, `<echelon>.backlog`, `<echelon>.expected`,
    and `<echelon>.supply_line` of the echelons `retailer`, `wholesaler`, `distributor`, and `factory`, and the flows
    `<upstream>_to_<downstream>.shipments` between the echelons, `<downstream>.arrivals` at every echelon a link
    supplies, `retailer.sales`, and `factory.starts` and `factory.completions`. The interactions `<echelon>_shipping`
    ship the backlog of every echelon and decay its expectation, the rates `<echelon>_backlog` and `<echelon>_expected`
    add the orders that reach every echelon and adapt its expectation, and the rates `<echelon>_supply_line` keep its
    supply line. The observables of the system are the orders `<echelon>.orders` and the shipments `<echelon>.shipments`
    of every echelon, and the orders of the customers `customers.orders`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import BeerGameSystem
        >>> [element.label for element in BeerGameSystem().prepare().system_.elements]
        ['retailer', 'wholesaler', 'distributor', 'factory']
    """

    field = 'economics/markets'
    topics = ('supply chain', 'bullwhip effect', 'delay', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Sterman 1989',)
    until = 36.0
    dt = 0.5
    observables = (
        *(Observable('orders', BreweryInteraction._get_orders, element=label) for label in ECHELONS),
        *(Observable('shipments', EchelonElement._get_shipments, element=label) for label in ECHELONS),
        Observable('customers.orders', _get_customer_orders),
    )

    @staticmethod
    def _get_received(downstream: SimpleNamespace) -> Expression:
        """Return the orders of the downstream echelon that reach its supplier.

        They arrive after a third-order mail delay.
        """
        return distributed_delay(BreweryInteraction._get_orders(downstream), order=3, time=downstream.mail_delay)

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.inventory = inventory
        self.demand = demand
        self.demand_step = demand_step
        self.step_time = step_time
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.dispatch_time = dispatch_time
        self.mail_delay = mail_delay
        self.shipping_delay = shipping_delay
        self.production_delay = production_delay

    @staticmethod
    def _get_link_rates(downstream: str, upstream: str) -> list[Interaction]:
        """Return the rates of a link.

        They are the supply line of the downstream echelon and the incoming orders upstream.
        """
        return [
            Interaction(
                f'{downstream}_supply_line',
                roles={'target': downstream, 'source': upstream},
                laws=lambda s: [
                    RateLaw(
                        s.target.supply_line,
                        BreweryInteraction._get_orders(s.target)
                        - SupplyLinkInteraction._get_arrivals(s.source, s.target),
                    ),
                ],
            ),
            Interaction(
                f'{upstream}_backlog',
                roles={'target': upstream, 'source': downstream},
                laws=lambda s: [RateLaw(s.target.backlog, BeerGameSystem._get_received(s.source))],
            ),
            Interaction(
                f'{upstream}_expected',
                roles={'target': upstream, 'source': downstream},
                laws=lambda s: [RateLaw(s.target.expected, s.target.theta * BeerGameSystem._get_received(s.source))],
            ),
        ]

    @staticmethod
    def _get_customer_rates() -> list[Interaction]:
        """Return the rates of the customers.

        They are the demand that reaches the retailer and its expectation.
        """
        return [
            Interaction(
                'retailer_backlog',
                roles={'target': 'retailer'},
                laws=lambda s: [RateLaw(s.target.backlog, _get_demand(s.target, s.t))],
            ),
            Interaction(
                'retailer_expected',
                roles={'target': 'retailer'},
                laws=lambda s: [RateLaw(s.target.expected, s.target.theta * _get_demand(s.target, s.t))],
            ),
        ]

    @staticmethod
    def _get_brewery_rate() -> Interaction:
        """Return the rate of the supply line of the factory.

        It is its orders less the brews completed.
        """
        return Interaction(
            'factory_supply_line',
            roles={'target': 'factory'},
            laws=lambda s: [
                RateLaw(
                    s.target.supply_line,
                    BreweryInteraction._get_orders(s.target) - BreweryInteraction._get_completions(s.target),
                ),
            ],
        )

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        throughput = self.demand * self.dispatch_time
        echelons: list[EchelonElement] = []
        for label in ECHELONS:
            pipeline = self.production_delay if label == 'factory' else self.shipping_delay
            lead_time = pipeline if label == 'factory' else self.mail_delay + self.shipping_delay
            supply_line = self.demand * lead_time
            values = {
                'inventory': self.inventory,
                'in_transit': self.demand * pipeline,
                'backlog': throughput,
                'expected': self.demand,
                'supply_line': supply_line,
                'desired': self.inventory - throughput + self.beta * supply_line,
                'theta': self.theta,
                'alpha': self.alpha,
                'beta': self.beta,
                'dispatch_time': self.dispatch_time,
                'mail_delay': self.mail_delay,
                'shipping_delay': self.shipping_delay,
            }
            if label == 'retailer':
                values.update(demand=self.demand, demand_step=self.demand_step, step_time=self.step_time)
            elif label == 'factory':
                values.update(production_delay=self.production_delay)
            kind = {'retailer': RetailerElement, 'factory': FactoryElement}.get(label, EchelonElement)
            echelons.append(kind(label).set_params(**values))
        shipping = [EchelonInteraction(f'{label}_shipping', target=label) for label in ECHELONS]
        links = [
            interaction
            for downstream, upstream in pairwise(ECHELONS)
            for interaction in (
                *self._get_link_rates(downstream, upstream),
                SupplyLinkInteraction(f'{upstream}_to_{downstream}', downstream=downstream, upstream=upstream),
            )
        ]
        customers = [*self._get_customer_rates(), RetailDemandInteraction('customers', retailer='retailer')]
        brewery = [self._get_brewery_rate(), BreweryInteraction('brewery', factory='factory')]
        self.system_ = System(self.label, elements=echelons, interactions=[*shipping, *links, *customers, *brewery])
