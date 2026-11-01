"""Stock-flow consistent monetary economy of households, firms, and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import TransferInteraction
from ....core.laws import BaseLaw, Sink, Source, UpdateLaw
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class HouseholdsElement(Element):
    """Households sector, whose deposits are money.

    Args:
        label:
            The label of the sector.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        deposits:
            The deposits of the households.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import HouseholdsElement
        >>> HouseholdsElement('households').deposits
        60.0
    """

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
    """Firms sector, with its deposits, its loans, and the sales of the last period.

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

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import FirmsElement
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
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits
        self.loans = loans
        self.sales = sales


class MonetaryBankElement(Element):
    """Bank of a stock-flow consistent economy, whose own funds are money.

    Args:
        label:
            The label of the bank.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        deposits:
            The own funds of the bank.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import MonetaryBankElement
        >>> MonetaryBankElement('bank').deposits
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        deposits: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='money')] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.deposits = deposits


class CircuitInteraction(TransferInteraction):
    """Circuit of money between households, firms, and a bank, one period per step.

    The flows of a period are the legs of one transfer between the deposits of the sectors, consumption, wages,
    interest, dividends, and profits, lending from the environment, and repayments to it, while the loans and the
    sales of the firms follow as maps. Every leg is named by its flow, so the reporters `<label>.<flow>` record it.

    Args:
        label:
            The label of the circuit.

        households:
            The path of the households.

        firms:
            The path of the firms.

        bank:
            The path of the bank.

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
            The share of the firms' deposits distributed as profits per period.

        credit:
            The new loans per unit of sales.

        repayment:
            The share of loans repaid per period.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CircuitInteraction
        >>> CircuitInteraction('circuit', households='households', firms='firms', bank='bank').credit
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        households: Annotated[str, RoleSpec()],
        firms: Annotated[str, RoleSpec()],
        bank: Annotated[str, RoleSpec()],
        alpha_income: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        alpha_wealth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        wage_share: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.7,
        interest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        payout: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        distribution: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        credit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        repayment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
    ) -> None:
        super().__init__(
            label,
            source=households,
            target=firms,
            state='households.deposits',
            into='firms.deposits',
            amount=self._get_consumption,
            flow='consumption',
            legs=[
                ('firms.deposits', 'households.deposits', lambda s: s.wage_share * s.firms.sales, 'wages'),
                ('firms.deposits', 'bank.deposits', lambda s: s.interest * s.firms.loans, 'interest'),
                ('bank.deposits', 'households.deposits', lambda s: s.payout * s.bank.deposits, 'dividends'),
                ('firms.deposits', 'households.deposits', lambda s: s.distribution * s.firms.deposits, 'profits'),
                (Source('lending'), 'firms.deposits', self._get_lending, 'lending'),
                ('firms.deposits', Sink('repayment'), lambda s: s.repayment * s.firms.loans, 'repayments'),
            ],
        )
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

    @staticmethod
    def _get_consumption(s: SimpleNamespace) -> Expression:
        """Return the consumption of a period, a share of the wages and a share of the deposits of the households."""
        return s.alpha_income * s.wage_share * s.firms.sales + s.alpha_wealth * s.households.deposits

    @staticmethod
    def _get_lending(s: SimpleNamespace) -> Expression:
        """Return the new loans of a period, in proportion to the sales of the firms."""
        return s.credit * s.firms.sales

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return the households and the firms, the first pair of the circuit, whose other legs name their roles."""
        return [(namespace.households, namespace.firms)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the transfers of one period and the maps of the loans and the sales.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the circuit.
        """
        firms = namespace.firms
        return [
            *super()._get_laws(namespace),
            UpdateLaw(firms.loans, firms.loans + self._get_lending(namespace) - namespace.repayment * firms.loans),
            UpdateLaw(firms.sales, self._get_consumption(namespace)),
        ]


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
