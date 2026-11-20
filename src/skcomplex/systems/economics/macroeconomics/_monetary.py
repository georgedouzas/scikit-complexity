"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Quantity
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class FirmsElement(Element):
    """Firms sector, with its deposits, its loans, and the sales of the last period.

    It pays wages and profits.

    Args:
        label:
            The label of the sector.

        deposits:
            The deposits of the firms, which are money.

        loans:
            The loans of the firms.

        sales:
            The sales of the last period.

        wage_share:
            The share of sales paid as wages.

        distribution:
            The share of the firms' deposits distributed as profits per period.

        repayment:
            The share of loans repaid per period.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import FirmsElement
        >>> FirmsElement('firms').loans
        50.0
    """

    def __init__(
        self,
        label: str,
        *,
        deposits: Annotated[object, RealSet.closed_open(0, oo), Quantity('money')] = 20.0,
        loans: Annotated[object, RealSet.closed_open(0, oo)] = 50.0,
        sales: Annotated[object, RealSet.closed_open(0, oo)] = 40.0,
        wage_share: Annotated[float, RealSet.closed_open(0, oo)] = 0.7,
        distribution: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        repayment: Annotated[float, RealSet.closed_open(0, oo)] = 0.08,
    ) -> None:
        super().__init__(label)
        self.deposits = deposits
        self.loans = loans
        self.sales = sales
        self.wage_share = wage_share
        self.distribution = distribution
        self.repayment = repayment


class HouseholdsElement(Element):
    """Households sector, whose deposits are money.

    It consumes out of its wages and its deposits.

    Args:
        label:
            The label of the sector.

        deposits:
            The deposits of the households.

        alpha_income:
            The propensity to consume out of wages.

        alpha_wealth:
            The propensity to consume out of deposits.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import HouseholdsElement
        >>> HouseholdsElement('households').deposits
        60.0
    """

    def __init__(
        self,
        label: str,
        *,
        deposits: Annotated[object, RealSet.closed_open(0, oo), Quantity('money')] = 60.0,
        alpha_income: Annotated[float, RealSet.closed_open(0, oo)] = 0.6,
        alpha_wealth: Annotated[float, RealSet.closed_open(0, oo)] = 0.4,
    ) -> None:
        super().__init__(label)
        self.deposits = deposits
        self.alpha_income = alpha_income
        self.alpha_wealth = alpha_wealth


class MonetaryBankElement(Element):
    """Bank of a stock-flow consistent economy, whose own funds are money.

    It lends and charges interest.

    Args:
        label:
            The label of the bank.

        deposits:
            The own funds of the bank.

        interest:
            The interest rate on loans per period.

        payout:
            The share of the bank's funds paid out as dividends per period.

        credit:
            The new loans per unit of sales.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import MonetaryBankElement
        >>> MonetaryBankElement('bank').deposits
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        deposits: Annotated[object, RealSet.closed_open(0, oo), Quantity('money')] = 10.0,
        interest: Annotated[float, RealSet.closed_open(0, oo)] = 0.03,
        payout: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        credit: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.deposits = deposits
        self.interest = interest
        self.payout = payout
        self.credit = credit


class CircuitInteraction(Interaction):
    """Circuit of money between households, firms, and a bank, one period per step.

    The flows of a period between the deposits of the sectors are consumption, wages, interest, dividends, and profits,
    lending from the environment, and repayments to it, each reading the parameters of the sectors. Every flow is
    labelled, so the trajectory records it.

    Args:
        label:
            The label of the circuit.

        households:
            The path of the households.

        firms:
            The path of the firms.

        bank:
            The path of the bank.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CircuitInteraction
        >>> CircuitInteraction('circuit', households='households', firms='firms', bank='bank').bank
        'bank'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, households: Role, firms: Role, bank: Role) -> None:
        super().__init__(label)
        self.households = households
        self.firms = firms
        self.bank = bank

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the flows of a period between the deposits of the sectors as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            households, firms, bank = context.households, context.firms, context.bank
            flows = [
                (
                    {households.deposits: -1, firms.deposits: +1},
                    self._get_consumption(households, firms),
                    'consumption',
                ),
                ({firms.deposits: -1, households.deposits: +1}, firms.wage_share * firms.sales, 'wages'),
                ({firms.deposits: -1, bank.deposits: +1}, bank.interest * firms.loans, 'interest'),
                ({bank.deposits: -1, households.deposits: +1}, bank.payout * bank.deposits, 'dividends'),
                ({firms.deposits: -1, households.deposits: +1}, firms.distribution * firms.deposits, 'profits'),
                ({firms.deposits: +1}, self._get_lending(firms, bank), 'lending'),
                ({firms.deposits: -1}, firms.repayment * firms.loans, 'repayments'),
            ]
            self.laws_: list[BaseComponent] = [RateLaw(target, amount, label=flow) for target, amount, flow in flows]

    @staticmethod
    def _get_consumption(households: SimpleNamespace, firms: SimpleNamespace) -> Expression:
        """Return the consumption of a period.

        It is a share of the wages and a share of the deposits of the households.
        """
        return households.alpha_income * firms.wage_share * firms.sales + households.alpha_wealth * households.deposits

    @staticmethod
    def _get_lending(firms: SimpleNamespace, bank: SimpleNamespace) -> Expression:
        """Return the new loans of a period, in proportion to the sales of the firms."""
        return bank.credit * firms.sales


class StockFlowConsistentEconomySystem(BaseSystem):
    """A stock-flow consistent economy.

    It is a closed circuit of money between households, firms, and a bank. Money is the deposits of the three sectors,
    tagged as the quantity `money`, and every transaction of a period is a flow, so each payment is one sector's outflow
    and another's inflow. Households consume a share `alpha_income` of their wages and a share `alpha_wealth` of their
    deposits, firms pay the wage share of last period's sales and the interest on their loans, and distribute a share of
    their deposits as profits, and the bank pays out a share of its funds as dividends. Lending creates money and
    repayment destroys it, the only flows with the environment, so the total money changes exactly by lending minus
    repayment, in the spirit of the systems of Godley and Lavoie (2007) and the balance sheets of Caiani et al. (2016).
    Loans approach `credit * sales / repayment` and sales settle where the stocks stop growing. The circuit moves the
    money of a period, and the interactions `loans` and `sales` set the loans and the sales of the firms.

    The system has the deposits `households.deposits`, `firms.deposits`, and `bank.deposits`, the loans `firms.loans`,
    the sales `firms.sales`, the flows `circuit.consumption`, `circuit.wages`, `circuit.interest`, `circuit.dividends`,
    and `circuit.profits` between the sectors, the flows `firms.lending` and `firms.repayments` with the environment,
    and the net money created `__boundary.money`.

    Args:
        household_deposits:
            The initial deposits of the households.

        firm_deposits:
            The initial deposits of the firms.

        bank_deposits:
            The initial own funds of the bank.

        loans:
            The initial loans of the firms.

        sales:
            The sales of the previous period.

        alpha_income:
            The propensity to consume out of wages.

        alpha_wealth:
            The propensity to consume out of deposits.

        wage_share:
            The share of sales paid as wages.

        interest:
            The interest rate on loans per period.

        payout:
            The share of the bank's funds paid out as dividends per period.

        distribution:
            The share of the firms' deposits distributed to households as profits per period.

        credit:
            The new loans per unit of sales.

        repayment:
            The share of loans repaid per period.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the Euler rule with a step of one period, so every flow moves
            the money of one period.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import StockFlowConsistentEconomySystem
        >>> [element.label for element in StockFlowConsistentEconomySystem().prepare().system_.elements]
        ['households', 'firms', 'bank']
    """

    field = 'economics/macroeconomics'
    topics = ('stocks and flows', 'stock-flow consistent', 'money', 'discrete map')
    references = ('Godley and Lavoie 2007', 'Caiani et al. 2016')
    until = 100
    dt = 1

    def __init__(
        self,
        *,
        household_deposits: float = 60.0,
        firm_deposits: float = 20.0,
        bank_deposits: float = 10.0,
        loans: float = 50.0,
        sales: float = 40.0,
        alpha_income: float = 0.6,
        alpha_wealth: float = 0.4,
        wage_share: float = 0.7,
        interest: float = 0.03,
        payout: float = 0.5,
        distribution: float = 0.3,
        credit: float = 0.1,
        repayment: float = 0.08,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.household_deposits = household_deposits
        self.firm_deposits = firm_deposits
        self.bank_deposits = bank_deposits
        self.loans = loans
        self.sales = sales
        self.alpha_income = alpha_income
        self.alpha_wealth = alpha_wealth
        self.wage_share = wage_share
        self.interest = interest
        self.payout = payout
        self.distribution = distribution
        self.credit = credit
        self.repayment = repayment

    @staticmethod
    def _get_next_loans(s: SimpleNamespace) -> list[BaseComponent]:
        """Return the loans of the firms after a period.

        The new loans are added and the repayments taken.
        """
        firms, bank = s.target, s.bank
        return [
            UpdateLaw(
                firms.loans,
                firms.loans + CircuitInteraction._get_lending(firms, bank) - firms.repayment * firms.loans,
            ),
        ]

    @staticmethod
    def _get_next_sales(s: SimpleNamespace) -> list[BaseComponent]:
        """Return the sales of the firms of a period, the consumption of the households."""
        return [UpdateLaw(s.target.sales, CircuitInteraction._get_consumption(s.households, s.target))]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[
                HouseholdsElement(
                    'households',
                    deposits=self.household_deposits,
                    alpha_income=self.alpha_income,
                    alpha_wealth=self.alpha_wealth,
                ),
                FirmsElement(
                    'firms',
                    deposits=self.firm_deposits,
                    loans=self.loans,
                    sales=self.sales,
                    wage_share=self.wage_share,
                    distribution=self.distribution,
                    repayment=self.repayment,
                ),
                MonetaryBankElement(
                    'bank',
                    deposits=self.bank_deposits,
                    interest=self.interest,
                    payout=self.payout,
                    credit=self.credit,
                ),
            ],
            interactions=[
                CircuitInteraction('circuit', households='households', firms='firms', bank='bank'),
                Interaction('loans', roles={'target': 'firms', 'bank': 'bank'}, laws=self._get_next_loans),
                Interaction('sales', roles={'target': 'firms', 'households': 'households'}, laws=self._get_next_sales),
            ],
            solver=EulerSolver(dt=1.0),
        )
