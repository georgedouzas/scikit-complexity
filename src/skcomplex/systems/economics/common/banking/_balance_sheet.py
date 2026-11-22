"""Bank of an interbank network.

It holds its external assets, its equity, its payments, and its state of default.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, Protocol

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element, VisualSpec
from .....core.components.types import AgentViewProtocol

SOLVENT = 0
DEFAULTING = 1
DEFAULTED = 2


class BalanceSheetView(AgentViewProtocol, Protocol):
    """Bank as a contagion sees it: its external assets, equity, payment, and status."""

    external: float
    equity: float
    payment: float
    status: float


class BalanceSheetElement(Element):
    """Bank of an interbank network.

    It holds its external assets, its equity, its payments, and its state of default. The bank is a passive agent that
    the contagion of the system updates, a solvency cascade through its equity or a clearing of payments through its
    payment.

    Args:
        label:
            The label of the agent or group.

        external:
            The external assets.

        equity:
            The equity, the assets minus the liabilities.

        payment:
            The payment the bank makes on its interbank liabilities.

        status:
            0 for solvent, 1 for defaulting in this step, and 2 for defaulted.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.finance import BalanceSheetElement
        >>> BalanceSheetElement('banks', equity=4.0, size=3).size
        3
    """

    visual = VisualSpec('agents', {'color_by': 'status'})

    def __init__(
        self,
        label: str,
        *,
        external: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        equity: object = 0.0,
        payment: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        status: Annotated[object, NN] = SOLVENT,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.external = external
        self.equity = equity
        self.payment = payment
        self.status = status
