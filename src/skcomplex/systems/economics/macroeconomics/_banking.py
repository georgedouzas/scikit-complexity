"""Macroeconomics systems: fractional-reserve banking.

Loans redeposited at banks multiply the base money.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Member
from ....core.components.interactions import Interaction, TransferInteraction
from ....core.components.laws import Sink, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import AgentView, BaseSolver, MesaSolver, SystemView
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.policy import PolicyRuleInteraction

TOLERANCE = 1e-9


class DepositorView(AgentView):
    """Customer as the rules see it.

    It holds the id of its `bank`, its `deposit`, its `loan`, and its shares of payments, the `spending` share of the
    deposit paid every step and the `repayment` share of the loan.
    """

    bank: int
    deposit: float
    loan: float
    spending: float
    repayment: float


class DepositorElement(Element):
    """Customer of a bank, with a deposit at the bank and a loan from it.

    Its value `bank` names the bank among the banks. The system pays a share of the deposit to a random customer, which
    moves deposits and as many reserves from the payer's bank to the payee's bank, and repays a share of the loan from
    the deposit, which cancels as much deposit and loan at the customer's bank and destroys broad money.

    Args:
        label:
            The label of the agent or group.

        bank:
            The index of the customer's bank among the banks, or -1 for none.

        deposit:
            The deposit at the bank.

        loan:
            The loan owed to the bank.

        spending:
            The share of the deposit paid to a random customer every step.

        repayment:
            The share of the loan repaid every step.

        size:
            The number of customers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import DepositorElement
        >>> DepositorElement('customers', deposit=5.0, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        bank: Annotated[object, Member('banks')] = -1,
        deposit: Annotated[object, RealSet.closed_open(0, oo)] = 10.0,
        loan: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        spending: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        repayment: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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
        """Return the repayment of a customer.

        It is a share `repayment` of its loan, at most its deposit.

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


class ReserveBankView(AgentView):
    """Bank as the rules see it: its `reserves`, `deposits`, `loans`, and `reserve_ratio`."""

    reserves: float
    deposits: float
    loans: float
    reserve_ratio: float


class ReserveBankElement(Element):
    """Bank with reserves, the deposits of its customers, and loans to them.

    The excess reserves are those beyond the required share `reserve_ratio` of its deposits, which `LendingInteraction`
    lends.

    Args:
        label:
            The label of the agent or group.

        reserves:
            The reserves of the bank.

        deposits:
            The deposits of its customers.

        loans:
            The loans to its customers.

        reserve_ratio:
            The required ratio of reserves to deposits.

        size:
            The number of banks, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import ReserveBankElement
        >>> ReserveBankElement('banks', reserves=10.0, size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        reserves: object = 0.0,
        deposits: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        loans: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        reserve_ratio: Annotated[object, RealSet.open(0, oo)] = 0.1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.reserves = reserves
        self.deposits = deposits
        self.loans = loans
        self.reserve_ratio = reserve_ratio

    @classmethod
    def _get_view(cls) -> type[ReserveBankView]:
        """Return the view of a bank, with its reserves, deposits, and loans."""
        return ReserveBankView


class LendingInteraction(Interaction):
    """Banks that lend their excess reserves to one of their customers.

    The excess reserves are those beyond the required share of a bank's deposits. The borrower spends the loan at once,
    and the payee, a random customer, deposits it at its own bank, whose reserves and deposits grow by the loan. The
    customers of a bank are those whose value `bank` names it.

    Args:
        label:
            The label of the interaction.

        target:
            The banks that lend.

        customers:
            The customers who borrow and are paid.

        phase:
            The phase the banks lend in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import LendingInteraction
        >>> LendingInteraction('lending', phase='lend').phase
        'lend'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'banks',
        customers: Role = 'customers',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.customers = customers
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the lending rule of every bank."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.lend, phase=self.phase)]

    def lend(self, agent: ReserveBankView, system: SystemView) -> None:
        """Lend the excess reserves to a random customer.

        The payment is redeposited at a random customer's bank.

        Args:
            agent:
                The bank.

            system:
                The system.
        """
        excess = agent.reserves - agent.reserve_ratio * agent.deposits
        customers = system.get_agents('customers', DepositorView)
        own = [customer for customer in customers if customer.get_referenced('bank') is agent]
        if excess <= TOLERANCE or not own:
            return
        borrower = own[int(system.rng.integers(len(own)))]
        payee = customers[int(system.rng.integers(len(customers)))]
        bank = payee.get_referenced('bank')
        if not isinstance(bank, ReserveBankView):
            return
        borrower.loan += excess
        agent.loans += excess
        agent.reserves -= excess
        payee.deposit += excess
        bank.deposits += excess
        bank.reserves += excess


class PolicySwitchInteraction(PolicyRuleInteraction[SystemView]):
    """Policy switch of a central bank.

    It moves the reserve ratio of the banks from `ratio` to `new_ratio`. The ratio switches at the time `change_at`.

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
        >>> from skcomplex.systems.economics.macroeconomics import PolicySwitchInteraction
        >>> PolicySwitchInteraction('regulator', ratio=0.1, new_ratio=0.2, change_at=100.0).new_ratio
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[SystemView], None] | None = None,
        phase: str | None = None,
        ratio: Annotated[float, RealSet.open(0, oo)] = 0.1,
        new_ratio: Annotated[float, RealSet.open(0, oo)] = 0.1,
        change_at: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        banks: str = 'banks',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.ratio = ratio
        self.new_ratio = new_ratio
        self.change_at = change_at
        self.banks = banks

    def act(self, system: SystemView) -> None:
        """Set the reserve ratio of the banks from the schedule.

        Args:
            system:
                The system view of the regulator.
        """
        before = system.time < system.get_parameter('change_at')
        ratio = system.get_parameter('ratio') if before else system.get_parameter('new_ratio')
        for bank in system.get_agents(self.banks, ReserveBankView):
            bank.reserve_ratio = ratio


class FractionalReserveBankingSystem(BaseSystem):
    """Customers and banks whose lending of excess reserves multiplies the base money.

    The multiplier is `1 / ratio`. The base money starts as equal deposits of the customers, held as reserves by their
    banks, picked at random. Every step the customers pay a share of their deposits to random customers and repay a
    share of their loans from their deposits, and every bank lends its reserves above the required share of its
    deposits. The loans are spent and redeposited at random banks. The reserves stay equal to the base money, so the
    broad money, the sum of deposits, settles near `base_money / ratio`. A regulator changes the reserve ratio to
    `new_ratio` at `change_at`, and the broad money moves to the new multiplier.

    The system has the groups `customers` and `banks`, the institution `regulator`, and the observables `broad_money`,
    `reserves`, `loans`, and `reserve_ratio`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import FractionalReserveBankingSystem
        >>> FractionalReserveBankingSystem(new_ratio=0.2).prepare().system_.interactions[0].new_ratio
        0.2
    """

    field = 'economics/macroeconomics'
    topics = ('money', 'banks', 'agents', 'institution')
    references = ('Hamill and Gilbert 2016 ch. 9',)
    until = 300
    observables = (
        Observable('broad_money', group='banks', variable='deposits', reduce='sum'),
        Observable('reserves', group='banks', variable='reserves', reduce='sum'),
        Observable('loans', group='banks', variable='loans', reduce='sum'),
        Observable('reserve_ratio', group='banks', variable='reserve_ratio', reduce='mean'),
    )

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.customers = customers
        self.banks = banks
        self.base_money = base_money
        self.ratio = ratio
        self.new_ratio = new_ratio
        self.change_at = change_at
        self.spending = spending
        self.repayment = repayment
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        owners = rng.integers(self.banks, size=self.customers)
        deposit = self.base_money / self.customers
        holdings = np.bincount(owners, minlength=self.banks) * deposit
        depositors = DepositorElement(
            'customers',
            bank=owners,
            deposit=deposit,
            spending=self.spending,
            repayment=self.repayment,
            size=self.customers,
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
            guard=lambda customer, same, system: DepositorElement.get_repayment(customer, same, system) > 0,
            form='agent',
            phase='pay',
        )
        lenders = ReserveBankElement(
            'banks',
            reserves=holdings,
            deposits=holdings,
            reserve_ratio=self.ratio,
            size=self.banks,
        )
        regulator = PolicySwitchInteraction(
            'regulator',
            phase='regulate',
            ratio=self.ratio,
            new_ratio=self.ratio if self.new_ratio is None else self.new_ratio,
            change_at=self.change_at,
        )
        self.system_ = System(
            self.label,
            elements=[depositors, lenders],
            interactions=[regulator, payments, repayments, LendingInteraction('lending', phase='lend')],
            solver=MesaSolver(phases=[PhaseSpec('regulate'), PhaseSpec('pay'), PhaseSpec('lend', repeat=10)]),
        )
