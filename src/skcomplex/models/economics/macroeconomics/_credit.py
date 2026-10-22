"""Macroeconomics models: credit networks of firms that borrow from banks and switch to cheaper lenders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.space import DiscreteTime, Phase, Phases
from ....core.system import Observable, System
from ..realizations import Market, SampleAndSwitch, Trade

type Array = npt.NDArray[np.float64]


def _lend(trade: Trade, model: Any) -> None:  # noqa: ANN401
    """Record a loan on the books of the firm and of the bank."""
    del model
    trade.buyer.loan += trade.quantity
    trade.buyer.rate = trade.price
    trade.seller.lent += trade.quantity


def _herfindahl(lent: Array) -> float:
    """Return the Herfindahl index of the banks' shares of the loans."""
    total = lent.sum()
    return float(np.sum((lent / total) ** 2)) if total > 0 else float('nan')


def _largest_share(lent: Array) -> float:
    """Return the share of the loans of the largest lender."""
    total = lent.sum()
    return float(lent.max() / total) if total > 0 else float('nan')


class Firm(Element):
    """Firm that borrows in proportion to its net worth, invests at a random return, and defaults on negative worth.

    Every step the firm asks for `leverage` times its net worth from its bank, invests its net worth and the loan at
    the return `mean_return + volatility * z`, and pays the interest. When its net worth becomes negative it repays
    what it can, the bank writes off the rest, and a new firm with the net worth `entry` replaces it.

    Args:
        label:
            The label of the agent or group.

        lenders:
            The path of the group of banks.

        net_worth:
            The initial net worth.

        demand:
            The loan still wanted in the current step.

        bid:
            The highest interest rate the firm pays.

        loan:
            The loan of the current step.

        rate:
            The interest rate of the loan, zero without a loan.

        partner:
            The id of the bank of the firm, `-1` before its first loan.

        gross_return:
            The return on the capital of the last step.

        injected:
            The net worth of an entrant that replaced the firm in the last step, zero otherwise.

        defaulted:
            Whether the firm defaulted in the last step.

        leverage:
            The ratio of the loan wanted to the net worth.

        mean_return:
            The mean return on capital.

        volatility:
            The standard deviation of the return on capital.

        entry:
            The net worth of an entrant.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Firm
        >>> Firm.group('firms', 10, net_worth=5.0).size
        10
    """

    lenders: str = 'banks'
    net_worth: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    demand: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    bid: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    loan: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    rate: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    partner: float = state(default=-1, domain=Domain.INTEGER)
    gross_return: float = state(default=0.0, domain=Domain.REAL)
    injected: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    defaulted: float = state(default=0, domain=Domain.BOOLEAN)
    leverage: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    mean_return: float = parameter(default=0.03, domain=Domain.REAL)
    volatility: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    entry: float = parameter(default=10.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the borrowing plan and the investment rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.plan, phase='plan'), Rule(self.invest, phase='settle')]

    def plan(self, agent: AgentView, model: ModelView) -> None:
        """Ask for a loan of `leverage` times the net worth.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        agent.demand = model.parameter('leverage') * agent.net_worth
        agent.loan, agent.rate, agent.defaulted, agent.injected = 0.0, 0.0, 0, 0.0

    def invest(self, agent: AgentView, model: ModelView) -> None:
        """Invest the capital, pay the interest to the bank, and default when the net worth turns negative.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        draw = model.parameter('mean_return') + model.parameter('volatility') * model.rng.standard_normal()
        capital = agent.net_worth + agent.loan
        agent.gross_return = capital * max(draw, -1.0)
        interest = agent.loan * agent.rate
        worth = agent.net_worth + agent.gross_return - interest
        if agent.loan > 0:
            bank = next(bank for bank in model.agents(self.lenders) if bank.id == int(agent.partner))
            bank.equity += interest + min(worth, 0.0)
            bank.losses += max(-worth, 0.0)
        if worth >= 0:
            agent.net_worth = worth
            return
        agent.net_worth = agent.injected = model.parameter('entry')
        agent.defaulted, agent.partner = 1, -1


class Bank(Element):
    """Bank that lends up to a multiple of its equity at a rate that falls with its equity.

    The bank asks the rate `alpha * equity ** -alpha` of Delli Gatti et al. (2010), so better capitalised banks lend
    cheaper, and it offers loans up to `equity / capital_ratio`.

    Args:
        label:
            The label of the agent or group.

        equity:
            The initial equity.

        ask:
            The interest rate the bank asks.

        supply:
            The loans the bank can still grant in the current step.

        lent:
            The loans granted in the current step.

        losses:
            The loans written off in the current step.

        alpha:
            The elasticity of the interest rate to the equity.

        capital_ratio:
            The smallest ratio of equity to loans.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Bank
        >>> Bank.group('banks', 5, equity=100.0).size
        5
    """

    equity: float = state(default=100.0, domain=Domain.REAL)
    ask: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    supply: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    lent: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    losses: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.1, domain=Domain.POSITIVE)
    capital_ratio: float = parameter(default=0.1, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the quoting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.quote, phase='plan')]

    def quote(self, agent: AgentView, model: ModelView) -> None:
        """Set the interest rate and the lending capacity from the equity.

        Args:
            agent:
                The bank.

            model:
                The model.
        """
        alpha = model.parameter('alpha')
        agent.ask = alpha * max(agent.equity, 1e-6) ** -alpha
        agent.supply = max(agent.equity, 0.0) / model.parameter('capital_ratio')
        agent.lent, agent.losses = 0.0, 0.0


class CreditMarket(Market):
    """Credit market that also reports the spread of the rates firms pay over the cheapest bank's rate.

    The spread is the mean of `r_i / r_min - 1` over the borrowing firms, the dispersion of rates that a market
    with costly search leaves and that switching to cheaper lenders erodes.

    Args:
        label:
            The label of the market.

        buyers:
            The path of the group of firms.

        sellers:
            The path of the group of banks.

        protocol:
            The protocol that matches firms with banks, such as `SampleAndSwitch()`.

        phase:
            The phase the market clears in, or `None` for the default phase.

        bid:
            The state of a firm with the highest rate it pays.

        ask:
            The state of a bank with the rate it asks.

        demand:
            The state of a firm with the loan it wants.

        supply:
            The state of a bank with the loans it can grant.

        money:
            The state that pays for trades, or `None` for no payment.

        goods:
            The state that holds the traded goods, or `None` for no delivery.

        on_trade:
            A function of every trade and the model view, run after the transfers, or `None`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CreditMarket
        >>> CreditMarket('credit', buyers='firms', sellers='banks').reports[-1]
        'spread'
    """

    reports: ClassVar[tuple[str, ...]] = (*Market.reports, 'spread')

    def clear(self, model: Any) -> None:  # noqa: ANN401
        """Clear the market and report the spread of the rates paid over the cheapest rate.

        Args:
            model:
                The model view of the market.
        """
        super().clear(model)
        cheapest = min((bank.ask for bank in model.agents('sellers') if bank.ask > 0), default=0.0)
        paid = [firm.rate for firm in model.agents('buyers') if firm.loan > 0]
        spread = float(np.mean(paid) / cheapest - 1) if paid and cheapest > 0 else float('nan')
        model.report('spread', spread)


def _credit_observables(system: System) -> list[Observable]:
    """Return the concentration, equity, worth, returns, injections, defaults, and losses of the credit market."""
    return [
        Observable('herfindahl', group='banks', variable='lent', reduce=_herfindahl),
        Observable('largest_share', group='banks', variable='lent', reduce=_largest_share),
        Observable('bank_equity', group='banks', variable='equity', reduce='sum'),
        Observable('firm_worth', group='firms', variable='net_worth', reduce='sum'),
        Observable('returns', group='firms', variable='gross_return', reduce='sum'),
        Observable('injected', group='firms', variable='injected', reduce='sum'),
        Observable('defaults', group='firms', variable='defaulted', reduce='sum'),
        Observable('bad_debt', group='banks', variable='losses', reduce='sum'),
    ]


@register_model(
    field='economics/macroeconomics',
    tags=('credit', 'banks', 'agents', 'market', 'network'),
    references=('Caiani et al. 2016 ch. 2', 'Delli Gatti et al. 2010'),
    defaults={'until': 100},
    observables=_credit_observables,
)
def credit_networks(
    firms: int = 100,
    banks: int = 10,
    *,
    sample: int = 3,
    intensity: float = 4.0,
    leverage: float = 1.0,
    mean_return: float = 0.03,
    volatility: float = 0.3,
    net_worth: float = 10.0,
    equity: float = 100.0,
    alpha: float = 0.1,
    capital_ratio: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a credit market where firms sample banks and switch to cheaper lenders, and defaults hit the lenders.

    Every step the banks quote rates that fall with their equity, every firm samples `sample` banks and switches from
    its lender to the cheapest of them with probability `1 - exp(intensity * (r_new - r_old) / r_new)`, borrows, and
    invests at a random return. Interest and write-offs of defaulted firms move net worth between firms and banks, so
    the total net worth changes only by the firms' returns and the entrants' net worth. A higher intensity of switching
    concentrates the loans in the cheapest banks and narrows the spread of the rates that firms pay.

    Args:
        firms:
            The number of firms.

        banks:
            The number of banks.

        sample:
            The number of banks every firm samples.

        intensity:
            The intensity of switching to a cheaper bank.

        leverage:
            The ratio of the loan a firm wants to its net worth.

        mean_return:
            The mean return on capital.

        volatility:
            The standard deviation of the return on capital.

        net_worth:
            The initial net worth of every firm and of every entrant.

        equity:
            The mean initial equity of the banks, drawn uniformly within 50% of it.

        alpha:
            The elasticity of the interest rate to the equity.

        capital_ratio:
            The smallest ratio of a bank's equity to its loans.

        random_state:
            The seed or generator of the initial equities.

    Returns:
        system:
            The model, with the groups `firms` and `banks`, the market `credit` with its reporter `credit.spread`, and
            the observables `herfindahl`, `largest_share`, `bank_equity`, `firm_worth`, `returns`, `injected`,
            `defaults`, and `bad_debt`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import credit_networks
        >>> [element.label for element in credit_networks().elements]
        ['firms', 'banks']
    """
    rng = np.random.default_rng(random_state)
    lenders = Bank.group(
        'banks',
        banks,
        equity=equity * rng.uniform(0.5, 1.5, banks),
        alpha=alpha,
        capital_ratio=capital_ratio,
    )
    borrowers = Firm.group(
        'firms',
        firms,
        net_worth=net_worth,
        leverage=leverage,
        mean_return=mean_return,
        volatility=volatility,
        entry=net_worth,
    )
    market = CreditMarket(
        'credit',
        buyers='firms',
        sellers='banks',
        protocol=SampleAndSwitch(sample=sample, intensity=intensity),
        phase='credit',
        money=None,
        goods=None,
        on_trade=_lend,
    )
    return System(
        components=[borrowers, lenders],
        interactions=[market],
        time=DiscreteTime(schedule=Phases([Phase('plan'), Phase('credit'), Phase('settle')])),
    )
