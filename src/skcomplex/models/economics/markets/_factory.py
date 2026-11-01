"""Factory of a supply chain, an echelon whose orders a brewery brews over the production delay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._echelon import EchelonElement


class FactoryElement(EchelonElement):
    """Factory of a supply chain, an echelon whose orders a brewery brews over the production delay.

    Args:
        label:
            The label of the factory.

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

        production_delay:
            The average time the brewery takes to brew an order, through a third-order delay.

    Examples:
        >>> from skcomplex.models.economics.markets import FactoryElement
        >>> FactoryElement('factory').production_delay
        3.0
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
        production_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
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
