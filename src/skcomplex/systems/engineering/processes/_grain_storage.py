"""Grain storage: a retail grain store whose stock oscillates.

Its orders through a supply line make the stock oscillate.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
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
        >>> GrainStorageSystem(receiving_delay=5.0).prepare().system_.elements[0].values['receiving_delay']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.stock = stock
        self.on_order = on_order
        self.desired_stock = desired_stock
        self.release_fraction = release_fraction
        self.receiving_delay = receiving_delay
        self.adjustment_delay = adjustment_delay

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the stocks."""
        return 'store'

    def _get_stocks(self) -> dict[str, object]:
        """Return the initial value of every stock by name."""
        return {
            'stock': self.stock,
            'on_order': self.on_order,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the stocks and parameters that have them, by name."""
        return {
            'desired_stock': RealSet.closed_open(0, oo),
            'release_fraction': RealSet.closed_open(0, oo),
            'receiving_delay': RealSet.open(0, oo),
            'adjustment_delay': RealSet.open(0, oo),
        }

    def _get_quantities(self) -> dict[str, str]:
        """Return the quantity every stock holds, by name."""
        return {
            'stock': 'grain',
            'on_order': 'grain',
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'desired_stock': self.desired_stock,
            'release_fraction': self.release_fraction,
            'receiving_delay': self.receiving_delay,
            'adjustment_delay': self.adjustment_delay,
        }

    def _get_flows(self) -> FlowTable:
        """Return the procurement, receiving, and release flows of grain."""
        return {
            'procurement': (
                None,
                'on_order',
                lambda s: (s.target.desired_stock - s.target.stock) / s.target.adjustment_delay,
            ),
            'receiving': ('on_order', 'stock', lambda s: s.target.on_order / s.target.receiving_delay),
            'release': ('stock', None, lambda s: s.target.stock * s.target.release_fraction),
        }
