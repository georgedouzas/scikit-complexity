"""Macroeconomics systems: fractional-reserve banking.

Loans redeposited at banks multiply the base money.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Member
from ....core.components.interactions import Interaction
from ....core.components.laws import PartnerSpec, UpdateLaw
from ....core.components.laws.utils import pairs
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.errors import SimulationError
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.policy import PolicyRuleInteraction

TOLERANCE = 1e-9


class DepositorView(AgentViewProtocol, Protocol):
    """Customer as the rules see it.

    It holds the id of its `bank`, its `deposit`, its `loan`, and its shares of payments, the `spending` share of the
    deposit paid every step and the `repayment` share of the loan.
    """

    bank: int
    deposit: float
    loan: float
    spending: float
    repayment: float


type Pairs = list[tuple[DepositorView, DepositorView]]


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


@runtime_checkable
class ReserveBankView(AgentViewProtocol, Protocol):
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


class PaymentsInteraction(Interaction):
    """Payments and repayments of the customers of the banks.

    Every customer pays a share `spending` of its deposit to a random customer, in random order, which moves the
    deposit and as many deposits and reserves from the payer's bank to the payee's bank. Every customer then repays a
    share `repayment` of its loan, at most its deposit, in random order, which cancels as much deposit and loan at the
    customer's bank and destroys broad money.

    Args:
        label:
            The label of the interaction.

        customers:
            The path of the group of customers.

        phase:
            The phase the customers pay in.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import PaymentsInteraction
        >>> PaymentsInteraction('payments', customers='customers').phase
        'pay'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, customers: Role, phase: str | None = 'pay') -> None:
        super().__init__(label)
        self.customers = customers
        self.phase = phase

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the payments and then the repayments of every step as rules of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [
            UpdateLaw(context, self.pay, phase=self.phase),
            UpdateLaw(context, self.repay, phase=self.phase),
        ]

    @staticmethod
    def _get_bank(customer: DepositorView) -> ReserveBankView:
        """Return the bank a customer names."""
        bank = customer.get_referenced('bank')
        if not isinstance(bank, ReserveBankView):
            error_msg = f'The customer {customer.id} names no bank.'
            raise SimulationError(error_msg)
        return bank

    def pay(self, system: SystemView) -> None:
        """Let every customer pay a share of its deposit to a random customer, in random order.

        Args:
            system:
                The system view of the payments.
        """
        customers = cast(list[DepositorView], system.get_agents('customers'))
        found = cast(Pairs, list(pairs(customers, customers, system.rng, PartnerSpec('all', 'uniform'))))
        for index in system.rng.permutation(len(found)):
            payer, payee = found[int(index)]
            if not (payer.alive and payee.alive):
                continue
            amount = payer.spending * payer.deposit
            paying, paid = self._get_bank(payer), self._get_bank(payee)
            payer.deposit -= amount
            payee.deposit += amount
            paying.deposits -= amount
            paid.deposits += amount
            paying.reserves -= amount
            paid.reserves += amount

    def repay(self, system: SystemView) -> None:
        """Let every customer repay a share of its loan from its deposit, in random order.

        Args:
            system:
                The system view of the repayments.
        """
        customers = cast(list[DepositorView], system.get_agents('customers'))
        for index in system.rng.permutation(len(customers)):
            customer = customers[int(index)]
            amount = min(customer.repayment * customer.loan, customer.deposit)
            if not customer.alive or amount <= 0:
                continue
            bank = self._get_bank(customer)
            customer.deposit -= amount
            customer.loan -= amount
            bank.deposits -= amount
            bank.loans -= amount


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

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the lending rule of every bank.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
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
        customers = cast(list[DepositorView], system.get_agents('customers'))
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
        for bank in cast(list[ReserveBankView], system.get_agents(self.banks)):
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
            interactions=[
                regulator,
                PaymentsInteraction('payments', customers='customers'),
                LendingInteraction('lending', phase='lend'),
            ],
            solver=MesaSolver(phases=[PhaseSpec('regulate'), PhaseSpec('pay'), PhaseSpec('lend', repeat=10)]),
        )
