"""Customer of a bank, with a deposit at the bank its reference state `bank` names and a loan from it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._depositor_view import DepositorView


class DepositorElement(Element):
    """Customer of a bank, with a deposit at the bank its reference state `bank` names and a loan from it.

    The system pays a share of the deposit to a random customer, which moves deposits and as many reserves from the
    payer's bank to the payee's bank, and repays a share of the loan from the deposit, which cancels as much deposit
    and loan at the customer's bank and destroys broad money.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        bank:
            The id of the customer's bank in the group `banks`, given initially as its position in the group.

        deposit:
            The deposit at the bank.

        loan:
            The loan owed to the bank.

        spending:
            The share of the deposit paid to a random customer every step.

        repayment:
            The share of the loan repaid every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import DepositorElement
        >>> GroupElement('customers', element=DepositorElement, size=10, values={'deposit': 5.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        bank: Annotated[float, StateSpec(domain=ZZ, reference='banks')] = -1,
        deposit: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        loan: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        spending: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        repayment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bank = bank
        self.deposit = deposit
        self.loan = loan
        self.spending = spending
        self.repayment = repayment

    @classmethod
    def _get_view(cls) -> type[DepositorView]:
        """Return the view of a customer, with its bank, deposit, and loan."""
        return DepositorView

    @staticmethod
    def get_payment(payer: DepositorView, payee: DepositorView, system: SystemView) -> float:
        """Return the payment of a customer to another, a share `spending` of its deposit.

        Args:
            payer:
                The customer who pays.

            payee:
                The customer who is paid.

            system:
                The system view of the payments.

        Returns:
            amount:
                The payment.
        """
        return payer.spending * payer.deposit

    @staticmethod
    def get_repayment(customer: DepositorView, same: DepositorView, system: SystemView) -> float:
        """Return the repayment of a customer, a share `repayment` of its loan, at most its deposit.

        Args:
            customer:
                The customer who repays.

            same:
                The customer again, the target of a transfer within members.

            system:
                The system view of the repayments.

        Returns:
            amount:
                The repayment.
        """
        return min(customer.repayment * customer.loan, customer.deposit)
