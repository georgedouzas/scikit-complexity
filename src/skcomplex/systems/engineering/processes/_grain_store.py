"""Retail grain store, whose stock is restocked from grain on order through a receiving delay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GrainStoreElement(Element):
    """Retail grain store, whose stock is restocked from grain on order through a receiving delay.

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
        """Return the procurement rate, the gap to the desired stock over the adjustment delay."""
        procurement: Expression = (s.desired_stock - s.stock) / s.adjustment_delay
        return procurement

    @staticmethod
    def _receiving(s: SimpleNamespace) -> Expression:
        """Return the receiving rate, the grain on order over the receiving delay."""
        receiving: Expression = s.on_order / s.receiving_delay
        return receiving
