"""Macroeconomics models: credit networks of firms that borrow from banks and switch to cheaper lenders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction, MatchingInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PartnerSwitchSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_model
from ._firm_view import FirmView
from ._lender_view import LenderView


def _lend(trade: MatchSpec, model: ModelView) -> None:
    """Record a loan on the books of the firm and of the bank."""
    del model
    firm, bank = trade.requester, trade.provider
    if not isinstance(firm, FirmView) or not isinstance(bank, LenderView):
        error_msg = 'A loan is a trade of a firm with a bank.'
        raise TypeError(error_msg)
    firm.loan += trade.quantity
    firm.rate = trade.price
    bank.lent += trade.quantity


def _herfindahl(values: object) -> float:
    """Return the Herfindahl index of the banks' shares of the loans."""
    lent = np.asarray(values, dtype=float)
    total = lent.sum()
    return float(np.sum((lent / total) ** 2)) if total > 0 else float('nan')


def _largest_share(values: object) -> float:
    """Return the share of the loans of the largest lender."""
    lent = np.asarray(values, dtype=float)
    total = lent.sum()
    return float(lent.max() / total) if total > 0 else float('nan')


class FirmElement(Element):
    """Firm that borrows in proportion to its net worth, invests at a random return, and defaults on negative worth.

    Every step the firm asks for `leverage` times its net worth from its bank, invests its net worth and the loan at
    the return `mean_return + volatility * z`, and pays the interest. When its net worth becomes negative it repays
    what it can, the bank writes off the rest, and a new firm with the net worth `entry` replaces it.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.macroeconomics import FirmElement
        >>> GroupElement('firms', element=FirmElement, size=10, values={'net_worth': 5.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        net_worth: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        loan: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rate: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        partner: Annotated[float, StateSpec(domain=ZZ, reference='banks')] = -1,
        gross_return: Annotated[float, StateSpec()] = 0.0,
        injected: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        defaulted: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        leverage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mean_return: Annotated[float, ParameterSpec()] = 0.03,
        volatility: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        entry: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.net_worth = net_worth
        self.demand = demand
        self.bid = bid
        self.loan = loan
        self.rate = rate
        self.partner = partner
        self.gross_return = gross_return
        self.injected = injected
        self.defaulted = defaulted
        self.leverage = leverage
        self.mean_return = mean_return
        self.volatility = volatility
        self.entry = entry

    @classmethod
    def _get_view(cls) -> type[FirmView]:
        """Return the view of a firm, with its balance sheet and its loan."""
        return FirmView

    @staticmethod
    def get_return(agent: FirmView, model: ModelView) -> float:
        """Return the return on the net worth and the loan invested, at a normal rate of return of at least -1.

        Args:
            agent:
                The firm.

            model:
                The model view of the investment.

        Returns:
            gross_return:
                The return of the step.
        """
        draw = agent.mean_return + agent.volatility * model.rng.standard_normal()
        return (agent.net_worth + agent.loan) * max(draw, -1.0)

    @staticmethod
    def get_worth(agent: FirmView) -> float:
        """Return the net worth after the return and the interest, negative for a firm that defaults.

        Args:
            agent:
                The firm.

        Returns:
            worth:
                The net worth before a default.
        """
        return agent.net_worth + agent.gross_return - agent.loan * agent.rate

    @staticmethod
    def get_plan() -> tuple[tuple[str, Callable[[FirmView, ModelView], float]], ...]:
        """Return the states a firm sets in the plan of a step, a loan request of `leverage` times its net worth.

        Returns:
            plan:
                The name of every state and the function of its value, of the firm and the model view.
        """
        return (
            ('demand', lambda agent, model: agent.leverage * agent.net_worth),
            ('loan', lambda agent, model: 0.0),
            ('rate', lambda agent, model: 0.0),
            ('defaulted', lambda agent, model: 0.0),
            ('injected', lambda agent, model: 0.0),
        )

    @staticmethod
    def get_default() -> tuple[tuple[str, Callable[[FirmView, ModelView], float]], ...]:
        """Return the states a firm sets after its return, those of an entrant for a firm that defaults.

        Returns:
            default:
                The name of every state and the function of its value, of the firm and the model view.
        """
        worth = FirmElement.get_worth
        return (
            ('injected', lambda agent, model: agent.entry if worth(agent) < 0 else agent.injected),
            ('defaulted', lambda agent, model: 1.0 if worth(agent) < 0 else agent.defaulted),
            ('partner', lambda agent, model: -1.0 if worth(agent) < 0 else agent.partner),
            ('net_worth', lambda agent, model: agent.entry if worth(agent) < 0 else worth(agent)),
        )


class LenderElement(Element):
    """Bank that lends up to a multiple of its equity at a rate that falls with its equity.

    The bank asks the rate `alpha * equity ** -alpha` of Delli Gatti et al. (2010), so better capitalised banks lend
    cheaper, and it offers loans up to `equity / capital_ratio`. The maps of its settlement read the firms bound to
    their role `sources`, whose interest and defaults reach the bank they name in their reference state `partner`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.macroeconomics import LenderElement
        >>> GroupElement('banks', element=LenderElement, size=5, values={'equity': 100.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        equity: Annotated[float, StateSpec()] = 100.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lent: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        losses: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        capital_ratio: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.equity = equity
        self.ask = ask
        self.supply = supply
        self.lent = lent
        self.losses = losses
        self.alpha = alpha
        self.capital_ratio = capital_ratio

    @classmethod
    def _get_view(cls) -> type[LenderView]:
        """Return the view of a bank, with its equity, quote, and loans."""
        return LenderView

    @staticmethod
    def get_quote() -> tuple[tuple[str, Callable[[LenderView, ModelView], float]], ...]:
        """Return the states a bank sets in the plan of a step, its rate and lending capacity from its equity.

        Returns:
            quote:
                The name of every state and the function of its value, of the bank and the model view.
        """
        return (
            ('ask', lambda agent, model: agent.alpha * max(agent.equity, 1e-6) ** -agent.alpha),
            ('supply', lambda agent, model: max(agent.equity, 0.0) / agent.capital_ratio),
            ('lent', lambda agent, model: 0.0),
            ('losses', lambda agent, model: 0.0),
        )

    @staticmethod
    def _get_borrowers(agent: LenderView, model: ModelView) -> list[FirmView]:
        """Return the firms bound to the role `sources` that borrowed from a bank in the step."""
        firms = model.agents('sources', FirmView)
        return [firm for firm in firms if firm.loan > 0 and firm.referenced('partner') is agent]

    @staticmethod
    def get_income(agent: LenderView, model: ModelView) -> float:
        """Return the interest a bank earns in the step, less what its defaulting borrowers cannot repay.

        Args:
            agent:
                The bank.

            model:
                The model view of the settlement.

        Returns:
            income:
                The change of the equity.
        """
        borrowers = LenderElement._get_borrowers(agent, model)
        return sum(firm.loan * firm.rate + min(FirmElement.get_worth(firm), 0.0) for firm in borrowers)

    @staticmethod
    def get_losses(agent: LenderView, model: ModelView) -> float:
        """Return the loans of a bank written off in the step.

        Args:
            agent:
                The bank.

            model:
                The model view of the settlement.

        Returns:
            losses:
                The change of the losses.
        """
        borrowers = LenderElement._get_borrowers(agent, model)
        return sum(max(-FirmElement.get_worth(firm), 0.0) for firm in borrowers)


class CreditMarketInteraction(MatchingInteraction[ModelView]):
    """Credit market that also reports the spread of the rates firms pay over the cheapest bank's rate.

    The firms request loans and the banks provide them. The spread is the mean of `r_i / r_min - 1` over the borrowing
    firms, the dispersion of rates that a market with costly search leaves and that switching to cheaper lenders
    erodes.

    Args:
        label:
            The label of the market.

        requesters:
            The path of the group of firms.

        providers:
            The path of the group of banks.

        rule:
            The spec of the algorithm that matches firms with banks, such as `PartnerSwitchSpec()`.

        limit:
            The state with the rate of both sides, or a pair of the firms' and the banks' states.

        quantity:
            The state with the loan of both sides, or a pair of the firms' and the banks' states.

        budget:
            The state of a firm with what it can spend.

        transfers:
            The states every loan moves, each with the sign `-1` for a payment of the firm or `1` for a delivery to
            it.

        phase:
            The phase the market clears in, or `None` for the default phase.

        on_match:
            A function of every loan and the model view, run after the transfers, or `None`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CreditMarketInteraction
        >>> CreditMarketInteraction('credit', requesters='firms', providers='banks').reports[-1]
        'spread'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'spread')

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelView) -> None:
        """Report the price, the volume, the unmatched quantity, and the spread of the rates over the cheapest rate.

        Args:
            matches:
                The loans of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            model:
                The model view of the market.
        """
        super()._report_matches(matches, unmatched, model)
        cheapest = min((bank.ask for bank in model.agents('providers', LenderView) if bank.ask > 0), default=0.0)
        paid = [firm.rate for firm in model.agents('requesters', FirmView) if firm.loan > 0]
        spread = float(np.mean(paid) / cheapest - 1) if paid and cheapest > 0 else float('nan')
        model.report('spread', spread)


def _credit_observables(system: System) -> list[ObservableSpec]:
    """Return the concentration, equity, worth, returns, injections, defaults, and losses of the credit market."""
    return [
        ObservableSpec('herfindahl', group='banks', variable='lent', reduce=_herfindahl),
        ObservableSpec('largest_share', group='banks', variable='lent', reduce=_largest_share),
        ObservableSpec('bank_equity', group='banks', variable='equity', reduce='sum'),
        ObservableSpec('firm_worth', group='firms', variable='net_worth', reduce='sum'),
        ObservableSpec('returns', group='firms', variable='gross_return', reduce='sum'),
        ObservableSpec('injected', group='firms', variable='injected', reduce='sum'),
        ObservableSpec('defaults', group='firms', variable='defaulted', reduce='sum'),
        ObservableSpec('bad_debt', group='banks', variable='losses', reduce='sum'),
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
    lenders = GroupElement(
        'banks',
        element=LenderElement,
        size=banks,
        values={'equity': equity * rng.uniform(0.5, 1.5, banks), 'alpha': alpha, 'capital_ratio': capital_ratio},
    )
    borrowers = GroupElement(
        'firms',
        element=FirmElement,
        size=firms,
        values={
            'net_worth': net_worth,
            'leverage': leverage,
            'mean_return': mean_return,
            'volatility': volatility,
            'entry': net_worth,
        },
    )
    market = CreditMarketInteraction(
        'credit',
        requesters='firms',
        providers='banks',
        rule=PartnerSwitchSpec(sample=sample, intensity=intensity),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        phase='credit',
        on_match=_lend,
    )
    plan = [
        MapInteraction(f'{group}_plan_{name}', target=group, state=name, function=function, form='agent', phase='plan')
        for group, steps in (('firms', FirmElement.get_plan()), ('banks', LenderElement.get_quote()))
        for name, function in steps
    ]
    settlement = [
        MapInteraction(
            'returns',
            target='firms',
            state='gross_return',
            function=FirmElement.get_return,
            form='agent',
            phase='settle',
        ),
        *(
            MapInteraction(
                f'bank_{name}',
                target='banks',
                state=name,
                function=function,
                sources=('firms',),
                mode='increment',
                form='agent',
                phase='settle',
            )
            for name, function in (('equity', LenderElement.get_income), ('losses', LenderElement.get_losses))
        ),
        *(
            MapInteraction(
                f'firms_settle_{name}',
                target='firms',
                state=name,
                function=function,
                form='agent',
                phase='settle',
            )
            for name, function in FirmElement.get_default()
        ),
    ]
    return System(
        elements=[borrowers, lenders],
        interactions=[*plan, market, *settlement],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('plan'), PhaseSpec('credit'), PhaseSpec('settle')])),
    )
