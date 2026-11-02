"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import MapInteraction
from ....core.system import System
from ..._catalog import register_model
from ._circuit import CircuitInteraction
from ._firms_element import FirmsElement
from ._households import HouseholdsElement
from ._monetary_bank import MonetaryBankElement


def _next_loans(s: SimpleNamespace) -> Expression:
    """Return the loans of the firms after a period, the new loans added and the repayments taken."""
    firms, bank = s.target, s.sources[0]
    return firms.loans + CircuitInteraction._lending(firms, bank) - firms.repayment * firms.loans


def _next_sales(s: SimpleNamespace) -> Expression:
    """Return the sales of the firms of a period, the consumption of the households."""
    return CircuitInteraction._consumption(s.sources[0], s.target)


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
    Loans approach `credit * sales / repayment` and sales settle where the stocks stop growing. The circuit moves the
    money of a period, and the maps `loans` and `sales` set the loans and the sales of the firms.

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
            the maps `loans` and `sales`, and the net money created `__boundary.money`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import stock_flow_consistent_economy
        >>> [element.label for element in stock_flow_consistent_economy().elements]
        ['households', 'firms', 'bank']
    """
    return System(
        elements=[
            HouseholdsElement(
                'households',
                deposits=household_deposits,
                alpha_income=alpha_income,
                alpha_wealth=alpha_wealth,
            ),
            FirmsElement(
                'firms',
                deposits=firm_deposits,
                loans=loans,
                sales=sales,
                wage_share=wage_share,
                distribution=distribution,
                repayment=repayment,
            ),
            MonetaryBankElement('bank', deposits=bank_deposits, interest=interest, payout=payout, credit=credit),
        ],
        interactions=[
            CircuitInteraction('circuit', households='households', firms='firms', bank='bank'),
            MapInteraction('loans', target='firms', state='loans', function=_next_loans, sources=['bank']),
            MapInteraction('sales', target='firms', state='sales', function=_next_sales, sources=['households']),
        ],
    )
