"""Grain storage: a retail grain store whose stock oscillates.

Its orders through a supply line make the stock oscillate.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GrainStoreElement(Element):
    """Retail grain store, whose stock is restocked from grain on order.

    Grain on order arrives in the stock at the receiving rate `on_order / receiving_delay`, and the stock releases the
    fraction `release_fraction` of itself per unit time to the consumers. The procurement rate
    `(desired_stock - stock) / adjustment_delay` adds grain on order, and it is negative when the stock exceeds the
    desired stock, which cancels orders.

    Args:
        label:
            The label of the store.

        visual:
            How the store is drawn, or `None` for `default_visual_`.

        stock:
            The initial grain stock.

        on_order:
            The initial grain on order.

        desired_stock:
            The desired grain stock.

        release_fraction:
            The fraction of the stock released per unit time.

        receiving_delay:
            The average time for grain on order to arrive.

        adjustment_delay:
            The time to close the gap between the desired and the current stock.

    Examples:
        >>> from skcomplex.systems.engineering.processes import GrainStoreElement
        >>> GrainStoreElement('store').desired_stock
        6000.0
    """

    law_classes = (TransferLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stock: Annotated[float, StateSpec(quantity='grain')] = 1000.0,
        on_order: Annotated[float, StateSpec(quantity='grain')] = 10000.0,
        desired_stock: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6000.0,
        release_fraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        receiving_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        adjustment_delay: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stock = stock
        self.on_order = on_order
        self.desired_stock = desired_stock
        self.release_fraction = release_fraction
        self.receiving_delay = receiving_delay
        self.adjustment_delay = adjustment_delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the procurement, receiving, and release flows of grain.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            TransferLaw(Source('suppliers'), s.on_order, GrainStoreElement._procurement(s), label='procurement'),
            TransferLaw(s.on_order, s.stock, GrainStoreElement._receiving(s), label='receiving'),
            TransferLaw(s.stock, Sink('consumers'), s.stock * s.release_fraction, label='release'),
        ]

    @staticmethod
    def _procurement(s: SimpleNamespace) -> Expression:
        """Return the procurement rate.

        It is the gap to the desired stock over the adjustment delay.
        """
        procurement: Expression = (s.desired_stock - s.stock) / s.adjustment_delay
        return procurement

    @staticmethod
    def _receiving(s: SimpleNamespace) -> Expression:
        """Return the receiving rate, the grain on order over the receiving delay."""
        receiving: Expression = s.on_order / s.receiving_delay
        return receiving


class GrainStorageSystem(BaseSystem):
    """A retail grain store, whose stock oscillates.

    Its orders through a supply line make the stock oscillate. The grain stock gains the receiving rate
    `on_order / receiving_delay` and loses the release rate `release_fraction stock`, and the grain on order gains the
    procurement rate `(desired_stock - stock) / adjustment_delay` and loses the receiving rate. The orders ignore the
    grain already on order, so the supply line overshoots and the stock, the receiving rate, and the procurement
    oscillate in weeks before they settle, with the stock below the desired one at
    `desired_stock / (1 + release_fraction adjustment_delay)`, 5333 tonnes for the defaults, where the release balances
    the receiving. The defaults are those of the grain storage example of Bala, Arshad and Noh (2017), in tonnes and
    weeks.

    The system has the stocks `store.stock` and `store.on_order` in tonnes, and the flows `store.procurement`,
    `store.receiving`, and `store.release` in tonnes per week, over weeks.

    Args:
        stock:
            The initial grain stock.

        on_order:
            The initial grain on order.

        desired_stock:
            The desired grain stock.

        release_fraction:
            The fraction of the stock released per week.

        receiving_delay:
            The average time for grain on order to arrive, in weeks.

        adjustment_delay:
            The time to close the gap between the desired and the current stock, in weeks.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.processes import GrainStorageSystem
        >>> GrainStorageSystem(receiving_delay=5.0).prepare().system_.elements[0].receiving_delay
        5.0
    """

    field = 'engineering/processes'
    topics = ('agriculture', 'storage', 'stocks and flows', 'system dynamics', 'oscillation', 'ode')
    references = ('Bala et al. 2017',)
    until = 100.0
    dt = 0.25

    def __init__(
        self,
        stock: float = 1000.0,
        on_order: float = 10000.0,
        *,
        desired_stock: float = 6000.0,
        release_fraction: float = 0.025,
        receiving_delay: float = 10.0,
        adjustment_delay: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.stock = stock
        self.on_order = on_order
        self.desired_stock = desired_stock
        self.release_fraction = release_fraction
        self.receiving_delay = receiving_delay
        self.adjustment_delay = adjustment_delay

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        store = GrainStoreElement(
            'store',
            stock=self.stock,
            on_order=self.on_order,
            desired_stock=self.desired_stock,
            release_fraction=self.release_fraction,
            receiving_delay=self.receiving_delay,
            adjustment_delay=self.adjustment_delay,
        )
        return System(self.label, elements=[store])
