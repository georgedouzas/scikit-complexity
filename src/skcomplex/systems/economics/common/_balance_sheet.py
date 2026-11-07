"""Bank of an interbank network, with its external assets, its equity, its payments, and its state of default."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw
from ....core.engine.solvers import AgentView
from ....core.specs import StateSpec, VisualSpec

SOLVENT = 0


DEFAULTING = 1


DEFAULTED = 2


class BalanceSheetView(AgentView):
    """Bank as a contagion sees it: its external assets, equity, payment, and status."""

    external: float
    equity: float
    payment: float
    status: float


class BalanceSheetElement(Element):
    """Bank of an interbank network, with its external assets, its equity, its payments, and its state of default.

    The bank is a passive agent that the contagion of the system updates, a solvency cascade through its equity or a
    clearing of payments through its payment.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        external:
            The external assets.

        equity:
            The equity, the assets minus the liabilities.

        payment:
            The payment the bank makes on its interbank liabilities.

        status:
            0 for solvent, 1 for defaulting in this step, and 2 for defaulted.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import BalanceSheetElement
        >>> GroupElement('banks', element=BalanceSheetElement, size=3, values={'equity': 4.0}).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'status'}),
        *,
        external: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        equity: Annotated[float, StateSpec()] = 0.0,
        payment: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        status: Annotated[float, StateSpec(domain=NN)] = SOLVENT,
    ) -> None:
        super().__init__(label, visual=visual)
        self.external = external
        self.equity = equity
        self.payment = payment
        self.status = status

    @classmethod
    def _get_view(cls) -> type[BalanceSheetView]:
        """Return the view of the agents of the element."""
        return BalanceSheetView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the contagion updates every bank at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
