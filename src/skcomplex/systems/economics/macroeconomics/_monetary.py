"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class FirmsElement(Element):
    """Firms sector, with its deposits, its loans, and the sales of the last period.

    It pays wages and profits.

    Args:
        label:
            The label of the sector.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 20.0,
        loans: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        sales: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        wage_share: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        distribution: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        repayment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
    ) -> None:
        super().__init__(label, visual=visual)
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

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 60.0,
        alpha_income: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        alpha_wealth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.alpha_income = alpha_income
        self.alpha_wealth = alpha_wealth


class MonetaryBankElement(Element):
    """Bank of a stock-flow consistent economy, whose own funds are money.

    It lends and charges interest.

    Args:
        label:
            The label of the bank.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 10.0,
        interest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        payout: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        credit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.interest = interest
        self.payout = payout
        self.credit = credit


class CircuitInteraction(TransferInteraction):
    """Circuit of money between households, firms, and a bank, one period per step.

    The flows of a period are the legs of one transfer between the deposits of the sectors, consumption, wages,
    interest, dividends, and profits, lending from the environment, and repayments to it, each reading the parameters
    of the sectors. Every leg is named by its flow, so the reporters `<label>.<flow>` record it.

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

    def __init__(
        self,
        label: str,
        *,
        households: Annotated[str, RoleSpec()],
        firms: Annotated[str, RoleSpec()],
        bank: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=households,
            target=firms,
            state='households.deposits',
            into='firms.deposits',
            amount=lambda s: CircuitInteraction._consumption(s.households, s.firms),
            flow='consumption',
            legs=[
                ('firms.deposits', 'households.deposits', lambda s: s.firms.wage_share * s.firms.sales, 'wages'),
                ('firms.deposits', 'bank.deposits', lambda s: s.bank.interest * s.firms.loans, 'interest'),
                ('bank.deposits', 'households.deposits', lambda s: s.bank.payout * s.bank.deposits, 'dividends'),
                (
                    'firms.deposits',
                    'households.deposits',
                    lambda s: s.firms.distribution * s.firms.deposits,
                    'profits',
                ),
                (
                    Source('lending'),
                    'firms.deposits',
                    lambda s: CircuitInteraction._lending(s.firms, s.bank),
                    'lending',
                ),
                ('firms.deposits', Sink('repayment'), lambda s: s.firms.repayment * s.firms.loans, 'repayments'),
            ],
        )
        self.households = households
        self.firms = firms
        self.bank = bank

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the households and the firms, the first pair of the circuit.

        The other legs name their roles.
        """
        return [(namespace.households, namespace.firms)]

    @staticmethod
    def _consumption(households: SimpleNamespace, firms: SimpleNamespace) -> Expression:
        """Return the consumption of a period.

        It is a share of the wages and a share of the deposits of the households.
        """
        return households.alpha_income * firms.wage_share * firms.sales + households.alpha_wealth * households.deposits

    @staticmethod
    def _lending(firms: SimpleNamespace, bank: SimpleNamespace) -> Expression:
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
    money of a period, and the maps `loans` and `sales` set the loans and the sales of the firms.

    The system has the deposits `households.deposits`, `firms.deposits`, and `bank.deposits`, the loans `firms.loans`,
    the sales `firms.sales`, the flows `circuit.consumption`, `circuit.wages`, `circuit.interest`, `circuit.dividends`,
    `circuit.profits`, `circuit.lending`, and `circuit.repayments`, the maps `loans` and `sales`, and the net money
    created `__boundary.money`.

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
    ) -> None:
        super().__init__(label)
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
    def _next_loans(s: SimpleNamespace) -> Expression:
        """Return the loans of the firms after a period.

        The new loans are added and the repayments taken.
        """
        firms, bank = s.target, s.sources[0]
        return firms.loans + CircuitInteraction._lending(firms, bank) - firms.repayment * firms.loans

    @staticmethod
    def _next_sales(s: SimpleNamespace) -> Expression:
        """Return the sales of the firms of a period, the consumption of the households."""
        return CircuitInteraction._consumption(s.sources[0], s.target)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        return System(
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
                MapInteraction('loans', target='firms', state='loans', function=self._next_loans, sources=['bank']),
                MapInteraction(
                    'sales',
                    target='firms',
                    state='sales',
                    function=self._next_sales,
                    sources=['households'],
                ),
            ],
        )
