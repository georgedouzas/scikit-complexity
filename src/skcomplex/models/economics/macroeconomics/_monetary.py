"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, Sink, Source, TransferLaw, UpdateLaw
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


class HouseholdsElement(Element):
    """Households element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 60.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits


class FirmsElement(Element):
    """Firms element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 20.0,
        loans: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        sales: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.loans = loans
        self.sales = sales


class MonetaryBankElement(Element):
    """Monetary bank element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits


def _circuit(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the transactions of one period between households, firms, and the bank."""
    households, firms, bank = s.households, s.firms, s.bank
    consumption = s.alpha_income * s.wage_share * firms.sales + s.alpha_wealth * households.deposits
    lending = s.credit * firms.sales
    return [
        TransferLaw(households.deposits, firms.deposits, consumption, label='consumption'),
        TransferLaw(firms.deposits, households.deposits, s.wage_share * firms.sales, label='wages'),
        TransferLaw(firms.deposits, bank.deposits, s.interest * firms.loans, label='interest'),
        TransferLaw(bank.deposits, households.deposits, s.payout * bank.deposits, label='dividends'),
        TransferLaw(firms.deposits, households.deposits, s.distribution * firms.deposits, label='profits'),
        TransferLaw(Source('lending'), firms.deposits, lending, label='lending'),
        TransferLaw(firms.deposits, Sink('repayment'), s.repayment * firms.loans, label='repayments'),
        UpdateLaw(firms.loans, firms.loans + lending - s.repayment * firms.loans),
        UpdateLaw(firms.sales, consumption),
    ]


class CircuitInteraction(Interaction):
    """Circuit interaction."""

    def __init__(
        self,
        label: str,
        *,
        households: Annotated[object, RoleSpec()],
        firms: Annotated[object, RoleSpec()],
        bank: Annotated[object, RoleSpec()],
        alpha_income: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        alpha_wealth: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        wage_share: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        interest: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        payout: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        distribution: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        credit: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        repayment: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
    ) -> None:
        super().__init__(label)
        self.households = households
        self.firms = firms
        self.bank = bank
        self.alpha_income = alpha_income
        self.alpha_wealth = alpha_wealth
        self.wage_share = wage_share
        self.interest = interest
        self.payout = payout
        self.distribution = distribution
        self.credit = credit
        self.repayment = repayment

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the circuit interaction."""
        return _circuit(s)


@register_model(
    field='economics/macroeconomics',
    tags=('stocks and flows', 'stock-flow consistent', 'money', 'discrete map'),
    references=('Godley and Lavoie 2007', 'Caiani et al. 2016'),
    defaults={'until': 100, 'dt': 1},
)
def stock_flow_consistent_economy(
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
) -> System:
    """Return a stock-flow consistent economy, a closed circuit of money between households, firms, and a bank.

    Money is the deposits of the three sectors, tagged as the quantity `money`, and every transaction of a period is
    a flow, so each payment is one sector's outflow and another's inflow. Households consume a share `alpha_income` of
    their wages and a share `alpha_wealth` of their deposits, firms pay the wage share of last period's sales and the
    interest on their loans, and distribute a share of their deposits as profits, and the bank pays out a share of its
    funds as dividends. Lending creates money and
    repayment destroys it, the only flows with the environment, so the total money changes exactly by lending minus
    repayment, in the spirit of the models of Godley and Lavoie (2007) and the balance sheets of Caiani et al. (2016).
    Loans approach `credit * sales / repayment` and sales settle where the stocks stop growing.

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

    Returns:
        system:
            The model, with the deposits `households.deposits`, `firms.deposits`, and `bank.deposits`, the loans
            `firms.loans`, the sales `firms.sales`, the flows `circuit.consumption`, `circuit.wages`,
            `circuit.interest`, `circuit.dividends`, `circuit.profits`, `circuit.lending`, and `circuit.repayments`,
            and the net money created `__boundary.money`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import stock_flow_consistent_economy
        >>> [element.label for element in stock_flow_consistent_economy().elements]
        ['households', 'firms', 'bank']
    """
    return System(
        elements=[
            HouseholdsElement('households', deposits=household_deposits),
            FirmsElement('firms', deposits=firm_deposits, loans=loans, sales=sales),
            MonetaryBankElement('bank', deposits=bank_deposits),
        ],
        interactions=[
            CircuitInteraction(
                'circuit',
                households='households',
                firms='firms',
                bank='bank',
                alpha_income=alpha_income,
                alpha_wealth=alpha_wealth,
                wage_share=wage_share,
                interest=interest,
                payout=payout,
                distribution=distribution,
                credit=credit,
                repayment=repayment,
            ),
        ],
    )
