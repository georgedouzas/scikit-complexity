"""Macroeconomics models: fractional-reserve banking, where loans redeposited at banks multiply the base money."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import FeedbackInteraction, TransferInteraction
from ....core.laws import BaseLaw, RuleLaw, Sink
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._depositor_view import DepositorView
from ._reserve_bank_view import ReserveBankView

TOLERANCE = 1e-9


class DepositorElement(Element):
    """Customer of a bank, with a deposit at the bank its reference state `bank` names and a loan from it.

    The model pays a share of the deposit to a random customer, which moves deposits and as many reserves from the
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
        >>> from skcomplex.models.economics.macroeconomics import DepositorElement
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
    def get_payment(payer: DepositorView, payee: DepositorView, model: ModelView) -> float:
        """Return the payment of a customer to another, a share `spending` of its deposit.

        Args:
            payer:
                The customer who pays.

            payee:
                The customer who is paid.

            model:
                The model view of the payments.

        Returns:
            amount:
                The payment.
        """
        return payer.spending * payer.deposit

    @staticmethod
    def get_repayment(customer: DepositorView, same: DepositorView, model: ModelView) -> float:
        """Return the repayment of a customer, a share `repayment` of its loan, at most its deposit.

        Args:
            customer:
                The customer who repays.

            same:
                The customer again, the target of a transfer within members.

            model:
                The model view of the repayments.

        Returns:
            amount:
                The repayment.
        """
        return min(customer.repayment * customer.loan, customer.deposit)


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
        >>> from skcomplex.models.economics.macroeconomics import ReserveBankElement
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

    def lend(self, agent: ReserveBankView, model: ModelView) -> None:
        """Lend the excess reserves to a random customer, whose payment is redeposited at a random customer's bank.

        Args:
            agent:
                The bank.

            model:
                The model.
        """
        excess = agent.reserves - model.parameter('reserve_ratio') * agent.deposits
        customers = model.agents(self.customers, DepositorView)
        own = [customer for customer in customers if customer.referenced('bank') is agent]
        if excess <= TOLERANCE or not own:
            return
        borrower = own[int(model.rng.integers(len(own)))]
        payee = customers[int(model.rng.integers(len(customers)))]
        bank = payee.referenced('bank')
        if not isinstance(bank, ReserveBankView):
            return
        borrower.loan += excess
        agent.loans += excess
        agent.reserves -= excess
        payee.deposit += excess
        bank.deposits += excess
        bank.reserves += excess


class PolicySwitchInteraction(FeedbackInteraction[ModelView]):
    """Policy switch of a central bank, which moves the reserve ratio of the banks from `ratio` to `new_ratio`.

    The ratio switches at the time `change_at`.

    Args:
        label:
            The label of the institution.

        rule:
            Unused, the regulator overrides `act`.

        phase:
            The phase the regulator acts in, or `None` for the default phase.

        ratio:
            The reserve ratio before the change.

        new_ratio:
            The reserve ratio after the change.

        change_at:
            The time of the change.

        banks:
            The path of the group of banks.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import PolicySwitchInteraction
        >>> PolicySwitchInteraction('regulator', ratio=0.1, new_ratio=0.2, change_at=100.0).new_ratio
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelView], None] | None = None,
        phase: str | None = None,
        ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        new_ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        change_at: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        banks: str = 'banks',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.ratio = ratio
        self.new_ratio = new_ratio
        self.change_at = change_at
        self.banks = banks

    def act(self, model: ModelView) -> None:
        """Set the reserve ratio of the banks from the schedule.

        Args:
            model:
                The model view of the regulator.
        """
        before = model.time < model.parameter('change_at')
        ratio = model.parameter('ratio') if before else model.parameter('new_ratio')
        name = f'{self.banks}.reserve_ratio'
        if model.parameters[name] != ratio:
            model.set_parameter(name, ratio)


@register_model(
    field='economics/macroeconomics',
    tags=('money', 'banks', 'agents', 'institution'),
    references=('Hamill and Gilbert 2016 ch. 9',),
    defaults={'until': 300},
    observables=lambda system: [
        ObservableSpec('broad_money', group='banks', variable='deposits', reduce='sum'),
        ObservableSpec('reserves', group='banks', variable='reserves', reduce='sum'),
        ObservableSpec('loans', group='banks', variable='loans', reduce='sum'),
    ],
)
def fractional_reserve_banking(
    customers: int = 100,
    banks: int = 5,
    *,
    base_money: float = 1000.0,
    ratio: float = 0.1,
    new_ratio: float | None = None,
    change_at: float = 150.0,
    spending: float = 0.1,
    repayment: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return customers and banks whose lending of excess reserves multiplies the base money by `1 / ratio`.

    The base money starts as equal deposits of the customers, held as reserves by their banks, picked at random. Every
    step the customers pay a share of their deposits to random customers and repay a share of their loans from their
    deposits, and every bank lends its reserves above the
    required share of its deposits; the loans are spent and redeposited at random banks. The reserves stay equal to the
    base money, so the broad money, the sum of deposits, settles near `base_money / ratio`. A regulator changes the
    reserve ratio to `new_ratio` at `change_at`, and the broad money moves to the new multiplier.

    Args:
        customers:
            The number of customers.

        banks:
            The number of banks.

        base_money:
            The base money, deposited by the customers in equal shares.

        ratio:
            The reserve ratio.

        new_ratio:
            The reserve ratio from `change_at` on, or `None` to keep `ratio`.

        change_at:
            The time the reserve ratio changes.

        spending:
            The share of its deposit a customer pays to a random customer every step.

        repayment:
            The share of its loan a customer repays every step.

        random_state:
            The seed or generator of the customers' banks.

    Returns:
        system:
            The model, with the groups `customers` and `banks`, the institution `regulator`, and the observables
            `broad_money`, `reserves`, and `loans`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import fractional_reserve_banking
        >>> fractional_reserve_banking(new_ratio=0.2).interactions[0].new_ratio
        0.2
    """
    rng = np.random.default_rng(random_state)
    owners = rng.integers(banks, size=customers)
    deposit = base_money / customers
    holdings = np.bincount(owners, minlength=banks) * deposit
    depositors = GroupElement(
        'customers',
        element=DepositorElement,
        size=customers,
        values={'bank': owners, 'deposit': deposit, 'spending': spending, 'repayment': repayment},
    )
    payments = TransferInteraction(
        'payments',
        source='customers',
        target='customers',
        state='deposit',
        amount=DepositorElement.get_payment,
        pairing='random',
        legs=(
            ('bank.deposits', None, DepositorElement.get_payment),
            ('bank.reserves', None, DepositorElement.get_payment),
        ),
        form='agent',
        phase='pay',
    )
    repayments = TransferInteraction(
        'repayments',
        source='customers',
        target='customers',
        state='deposit',
        into=Sink(),
        amount=DepositorElement.get_repayment,
        legs=tuple(
            (state, Sink(), DepositorElement.get_repayment) for state in ('loan', 'bank.deposits', 'bank.loans')
        ),
        guard=lambda customer, same, model: DepositorElement.get_repayment(customer, same, model) > 0,
        form='agent',
        phase='pay',
    )
    lenders = GroupElement(
        'banks',
        element=ReserveBankElement,
        size=banks,
        values={'reserves': holdings, 'deposits': holdings, 'reserve_ratio': ratio},
    )
    regulator = PolicySwitchInteraction(
        'regulator',
        phase='regulate',
        ratio=ratio,
        new_ratio=ratio if new_ratio is None else new_ratio,
        change_at=change_at,
    )
    return System(
        elements=[depositors, lenders],
        interactions=[regulator, payments, repayments],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('regulate'), PhaseSpec('pay'), PhaseSpec('lend', repeat=10)]),
        ),
    )
