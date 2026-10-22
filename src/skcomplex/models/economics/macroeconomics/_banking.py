"""Macroeconomics models: fractional-reserve banking, where loans redeposited at banks multiply the base money."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from ....core.base import Domain, parameter, state
from ....core.model import AgentType, DiscreteTime, Institution, Law, Observable, Phase, Phases, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

TOLERANCE = 1e-9


def _bank_of(banks: list[AgentView], index: float) -> AgentView:
    """Return the bank with an index."""
    return next(bank for bank in banks if int(bank.index) == int(index))


class Depositor(AgentType):
    """Customer who pays a share of its deposit to a random customer and repays a share of its loan every step.

    A payment moves deposits and as many reserves from the payer's bank to the payee's bank, and a repayment from the
    deposit cancels as much deposit and loan at the customer's bank, destroying broad money.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        lenders:
            The path of the group of banks.

        bank:
            The index of the customer's bank.

        deposit:
            The deposit at the bank.

        loan:
            The loan owed to the bank.

        spending:
            The share of the deposit paid to a random customer every step.

        repayment:
            The share of the loan repaid every step.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Depositor
        >>> Depositor.group('customers', 10, deposit=5.0).size
        10
    """

    lenders: str = 'banks'
    bank: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    deposit: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    loan: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    spending: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    repayment: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the payment and repayment rules.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.pay, phase='pay'), Rule(self.repay, phase='pay')]

    def pay(self, agent: AgentView, model: ModelView) -> None:
        """Pay a share of the deposit to a random customer, moving deposits and reserves between their banks.

        Args:
            agent:
                The customer.

            model:
                The model.
        """
        customers = model.agents(agent.group)
        payee = customers[int(model.rng.integers(len(customers)))]
        amount = model.parameter('spending') * agent.deposit
        banks = model.agents(self.lenders)
        source, target = _bank_of(banks, agent.bank), _bank_of(banks, payee.bank)
        agent.deposit -= amount
        payee.deposit += amount
        source.deposits -= amount
        source.reserves -= amount
        target.deposits += amount
        target.reserves += amount

    def repay(self, agent: AgentView, model: ModelView) -> None:
        """Repay a share of the loan from the deposit, which destroys as much broad money.

        Args:
            agent:
                The customer.

            model:
                The model.
        """
        amount = min(model.parameter('repayment') * agent.loan, agent.deposit)
        if amount <= 0:
            return
        bank = _bank_of(model.agents(self.lenders), agent.bank)
        agent.deposit -= amount
        agent.loan -= amount
        bank.deposits -= amount
        bank.loans -= amount


class ReserveBank(AgentType):
    """Bank that lends its reserves in excess of the required share of its deposits to one of its customers.

    The borrower spends the loan at once, and the payee, a random customer, deposits it at its own bank, whose
    reserves and deposits grow by the loan.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        customers:
            The path of the group of customers.

        index:
            The index of the bank.

        reserves:
            The reserves of the bank.

        deposits:
            The deposits of its customers.

        loans:
            The loans to its customers.

        reserve_ratio:
            The required ratio of reserves to deposits.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import ReserveBank
        >>> ReserveBank.group('banks', 3, index=[0, 1, 2]).size
        3
    """

    customers: str = 'customers'
    index: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    reserves: float = state(default=0.0, domain=Domain.REAL)
    deposits: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    loans: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    reserve_ratio: float = parameter(default=0.1, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the lending rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.lend, phase='lend')]

    def lend(self, agent: AgentView, model: ModelView) -> None:
        """Lend the excess reserves to a random customer, whose payment is redeposited at a random customer's bank.

        Args:
            agent:
                The bank.

            model:
                The model.
        """
        excess = agent.reserves - model.parameter('reserve_ratio') * agent.deposits
        customers = model.agents(self.customers)
        own = [customer for customer in customers if int(customer.bank) == int(agent.index)]
        if excess <= TOLERANCE or not own:
            return
        borrower = own[int(model.rng.integers(len(own)))]
        payee = customers[int(model.rng.integers(len(customers)))]
        bank = _bank_of(model.agents(agent.group), payee.bank)
        borrower.loan += excess
        agent.loans += excess
        agent.reserves -= excess
        payee.deposit += excess
        bank.deposits += excess
        bank.reserves += excess


class Regulator(Institution):
    """Central bank that sets the reserve ratio of the banks, changing it from `ratio` to `new_ratio` at `change_at`.

    Args:
        label:
            The label of the institution.

        rule:
            Unused, the regulator overrides `act`.

        phase:
            The phase the regulator acts in, or `None` for the default phase.

        banks:
            The path of the group of banks.

        ratio:
            The reserve ratio before the change.

        new_ratio:
            The reserve ratio after the change.

        change_at:
            The time of the change.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Regulator
        >>> Regulator('regulator', ratio=0.1, new_ratio=0.2, change_at=100.0).new_ratio
        0.2
    """

    banks: str = 'banks'
    ratio: float = parameter(default=0.1, domain=Domain.POSITIVE)
    new_ratio: float = parameter(default=0.1, domain=Domain.POSITIVE)
    change_at: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def act(self, model: Any) -> None:  # noqa: ANN401
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
        Observable('broad_money', group='banks', variable='deposits', reduce='sum'),
        Observable('reserves', group='banks', variable='reserves', reduce='sum'),
        Observable('loans', group='banks', variable='loans', reduce='sum'),
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
    depositors = Depositor.group(
        'customers',
        customers,
        bank=owners,
        deposit=deposit,
        spending=spending,
        repayment=repayment,
    )
    lenders = ReserveBank.group(
        'banks',
        banks,
        index=np.arange(banks),
        reserves=holdings,
        deposits=holdings,
        reserve_ratio=ratio,
    )
    regulator = Regulator(
        'regulator',
        phase='regulate',
        ratio=ratio,
        new_ratio=ratio if new_ratio is None else new_ratio,
        change_at=change_at,
    )
    return System(
        components=[depositors, lenders],
        interactions=[regulator],
        time=DiscreteTime(schedule=Phases([Phase('regulate'), Phase('pay'), Phase('lend', repeat=10)])),
    )
