"""Grain storage: a retail grain store whose stock oscillates.

Its orders through a supply line make the stock oscillate.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.laws import Sink, Source
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import StockFlowSystem
from ...general.dynamics.types import FlowTable


class GrainStorageSystem(StockFlowSystem):
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
        >>> GrainStorageSystem(receiving_delay=5.0).prepare().system_.elements[0].get_value('receiving_delay')
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
        super().__init__(label=label)
        self.stock = stock
        self.on_order = on_order
        self.desired_stock = desired_stock
        self.release_fraction = release_fraction
        self.receiving_delay = receiving_delay
        self.adjustment_delay = adjustment_delay

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks."""
        return 'store'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the stocks and the parameters with their specs."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'stock': DeclaredVariableSpec(self.stock, StateSpec(quantity='grain')),
            'on_order': DeclaredVariableSpec(self.on_order, StateSpec(quantity='grain')),
            'desired_stock': DeclaredVariableSpec(self.desired_stock, ParameterSpec(domain=nonnegative)),
            'release_fraction': DeclaredVariableSpec(self.release_fraction, ParameterSpec(domain=nonnegative)),
            'receiving_delay': DeclaredVariableSpec(self.receiving_delay, ParameterSpec(domain=RealSet.open(0, oo))),
            'adjustment_delay': DeclaredVariableSpec(self.adjustment_delay, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_flows(self) -> FlowTable:
        """Return the procurement, receiving, and release flows of grain."""
        return {
            'procurement': (
                Source('suppliers'),
                'on_order',
                lambda s: (s.source.desired_stock - s.source.stock) / s.source.adjustment_delay,
            ),
            'receiving': ('on_order', 'stock', lambda s: s.source.on_order / s.source.receiving_delay),
            'release': ('stock', Sink('consumers'), lambda s: s.source.stock * s.source.release_fraction),
        }
