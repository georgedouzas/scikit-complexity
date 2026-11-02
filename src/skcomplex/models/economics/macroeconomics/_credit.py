"""Macroeconomics models: credit networks of firms that borrow from banks and switch to cheaper lenders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    ObservableSpec,
    PartnerSwitchSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ..._catalog import register_model
from ._credit_market import CreditMarketInteraction
from ._firm import FirmElement
from ._firm_view import FirmView
from ._lender import LenderElement
from ._lender_view import LenderView


def _lend(trade: MatchSpec, model: ModelView) -> None:
    """Record a loan on the books of the firm and of the bank."""
    del model
    firm, bank = trade.requester, trade.provider
    if not isinstance(firm, FirmView) or not isinstance(bank, LenderView):
        error_msg = 'A loan is a trade of a firm with a bank.'
        raise TypeError(error_msg)
    firm.loan += trade.quantity
    firm.rate = trade.value
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
