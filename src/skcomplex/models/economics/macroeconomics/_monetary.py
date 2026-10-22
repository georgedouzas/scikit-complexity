"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Auxiliary, Element, Interaction, Law, Sink, Source, Symbols, System, Transfer, Update
from ....core.runtime import register_model

Households = Element.define(
    'Households',
    states={'deposits': state(default=60.0, domain=Domain.NONNEGATIVE, quantity='money')},
)

Firms = Element.define(
    'Firms',
    states={
        'deposits': state(default=20.0, domain=Domain.NONNEGATIVE, quantity='money'),
        'loans': state(default=50.0, domain=Domain.NONNEGATIVE),
        'sales': state(default=40.0, domain=Domain.NONNEGATIVE),
    },
)

MonetaryBank = Element.define(
    'MonetaryBank',
    states={'deposits': state(default=10.0, domain=Domain.NONNEGATIVE, quantity='money')},
)


def _circuit(s: Symbols) -> list[Law]:
    """Return the transactions of one period between households, firms, and the bank."""
    households, firms, bank = s.households, s.firms, s.bank
    consumption = Auxiliary(
        'consumption_demand',
        s.alpha_income * s.wage_share * firms.sales + s.alpha_wealth * households.deposits,
    )
    lending = Auxiliary('new_loans', s.credit * firms.sales)
    return [
        consumption,
        lending,
        Transfer(households.deposits, firms.deposits, consumption.symbol, label='consumption'),
        Transfer(firms.deposits, households.deposits, s.wage_share * firms.sales, label='wages'),
        Transfer(firms.deposits, bank.deposits, s.interest * firms.loans, label='interest'),
        Transfer(bank.deposits, households.deposits, s.payout * bank.deposits, label='dividends'),
        Transfer(firms.deposits, households.deposits, s.distribution * firms.deposits, label='profits'),
        Transfer(Source('lending'), firms.deposits, lending.symbol, label='lending'),
        Transfer(firms.deposits, Sink('repayment'), s.repayment * firms.loans, label='repayments'),
        Update(firms.loans, firms.loans + lending.symbol - s.repayment * firms.loans),
        Update(firms.sales, consumption.symbol),
    ]


Circuit = Interaction.define(
    'Circuit',
    roles=['households', 'firms', 'bank'],
    parameters={
        'alpha_income': parameter(default=0.6, domain=Domain.NONNEGATIVE),
        'alpha_wealth': parameter(default=0.4, domain=Domain.NONNEGATIVE),
        'wage_share': parameter(default=0.7, domain=Domain.NONNEGATIVE),
        'interest': parameter(default=0.03, domain=Domain.NONNEGATIVE),
        'payout': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'distribution': parameter(default=0.3, domain=Domain.NONNEGATIVE),
        'credit': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'repayment': parameter(default=0.08, domain=Domain.NONNEGATIVE),
    },
    laws=_circuit,
)


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
        components=[
            Households('households', deposits=household_deposits),
            Firms('firms', deposits=firm_deposits, loans=loans, sales=sales),
            MonetaryBank('bank', deposits=bank_deposits),
        ],
        interactions=[
            Circuit(
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
