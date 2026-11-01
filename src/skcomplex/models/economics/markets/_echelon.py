"""Echelon of a supply chain, which ships its backlog from its inventory and orders by anchoring and adjustment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import minimum
from ....specs import ParameterSpec, StateSpec, VisualSpec


class EchelonElement(Element):
    """Echelon of a supply chain, which ships its backlog from its inventory and orders by anchoring and adjustment.

    It ships its backlog over the dispatch time, limited by its inventory. Its expected demand adapts to incoming
    orders at the rate `theta`, and it orders its expected demand plus `alpha` times the gap between its desired stock
    and its effective inventory, the inventory less the backlog, less `beta` times its supply line, never negative.
    Its orders reach its supplier after the mail delay, and the beer its supplier ships arrives after the shipping
    delay.

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

        mail_delay:
            The average time an order of the echelon takes to reach its supplier.

        shipping_delay:
            The average time a shipment takes to reach the echelon from its supplier.

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
        mail_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        shipping_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
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
        self.mail_delay = mail_delay
        self.shipping_delay = shipping_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shipping of the backlog and the decay of the expected demand.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return EchelonElement._echelon_laws(s)

    @staticmethod
    def _shipments(echelon: SimpleNamespace) -> Expression:
        """Return the shipments of an echelon, its backlog limited by its inventory over the dispatch time."""
        return minimum(echelon.backlog, echelon.inventory) / echelon.dispatch_time

    @staticmethod
    def _echelon_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of an echelon, the shipping of its backlog and the decay of its expectation."""
        return [
            RateLaw(s.backlog, -EchelonElement._shipments(s)),
            RateLaw(s.expected, -s.theta * s.expected),
        ]
