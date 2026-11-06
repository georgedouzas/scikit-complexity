"""Bank that lends its reserves in excess of the required share of its deposits to one of its customers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._depositor_view import DepositorView
from ._reserve_bank_view import ReserveBankView

TOLERANCE = 1e-9


class ReserveBankElement(Element):
    """Bank that lends its reserves in excess of the required share of its deposits to one of its customers.

    The borrower spends the loan at once, and the payee, a random customer, deposits it at its own bank, whose
    reserves and deposits grow by the loan. The customers of a bank are those whose reference state `bank` names it.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        reserves:
            The reserves of the bank.

        deposits:
            The deposits of its customers.

        loans:
            The loans to its customers.

        reserve_ratio:
            The required ratio of reserves to deposits.

        customers:
            The path of the group of customers.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import ReserveBankElement
        >>> GroupElement('banks', element=ReserveBankElement, size=3, values={'reserves': 10.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        reserves: Annotated[float, StateSpec()] = 0.0,
        deposits: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        loans: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        reserve_ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        customers: str = 'customers',
    ) -> None:
        super().__init__(label, visual=visual)
        self.reserves = reserves
        self.deposits = deposits
        self.loans = loans
        self.reserve_ratio = reserve_ratio
        self.customers = customers

    @classmethod
    def _get_view(cls) -> type[ReserveBankView]:
        """Return the view of a bank, with its reserves, deposits, and loans."""
        return ReserveBankView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the lending rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.lend, phase='lend')]

    def lend(self, agent: ReserveBankView, system: SystemView) -> None:
        """Lend the excess reserves to a random customer, whose payment is redeposited at a random customer's bank.

        Args:
            agent:
                The bank.

            system:
                The system.
        """
        excess = agent.reserves - system.parameter('reserve_ratio') * agent.deposits
        customers = system.agents(self.customers, DepositorView)
        own = [customer for customer in customers if customer.referenced('bank') is agent]
        if excess <= TOLERANCE or not own:
            return
        borrower = own[int(system.rng.integers(len(own)))]
        payee = customers[int(system.rng.integers(len(customers)))]
        bank = payee.referenced('bank')
        if not isinstance(bank, ReserveBankView):
            return
        borrower.loan += excess
        agent.loans += excess
        agent.reserves -= excess
        payee.deposit += excess
        bank.deposits += excess
        bank.reserves += excess
