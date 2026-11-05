"""Market maker that clears the risky asset at the price where the mean forecast of the traders is met."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import FeedbackInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ..common.forecasting import heuristics
from ._heuristic_trader import PREDICTORS
from ._heuristic_trader_view import HeuristicTraderView
from ._market_state_spec import _MarketStateSpec


class HeuristicPricingInteraction(FeedbackInteraction[SystemView]):
    """Market maker that clears the risky asset at the price where the mean forecast of the traders is met.

    The deviation `x` of the price from its fundamental value clears at `R x_t = sum_h n_h f_h + eps_t`, where `n_h`
    is the share of traders using heuristic `h`. The fundamentalists forecast the fundamental, `f = 0`, and the trend
    chasers extrapolate the last deviation, `f = g x_{t-1}`, both as `heuristics.fundamental` with a reversion of 0
    and `g`. The realised fitness of a heuristic is the profit of its last forecast,
    `(x_t - R x_{t-1}) (f_h(t - 1) - R x_{t-1}) / (a sigma^2)`, minus its cost.

    Args:
        label:
            The label of the institution.

        rule:
            Unused, since the institution overrides `act`.

        phase:
            The phase the institution clears in.

        traders:
            The path of the group of traders.

        rate:
            The gross return `R` of the riskless asset.

        strength:
            The extrapolation `g` of the trend chasers.

        cost:
            The cost of the fundamentalist heuristic.

        risk:
            The risk aversion times the variance of returns, `a sigma^2`.

        noise:
            The standard deviation of the noise of the price.

        initial:
            The initial deviation of the price from its fundamental value.

    Examples:
        >>> from skcomplex.systems.economics.markets import HeuristicPricingInteraction
        >>> HeuristicPricingInteraction('pricing', traders='traders').reports
        ('deviation', 'trend_share', 'fitness_fundamentalist', 'fitness_trend')
    """

    reports: ClassVar[tuple[str, ...]] = ('deviation', 'trend_share', 'fitness_fundamentalist', 'fitness_trend')

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[SystemView], None] | None = None,
        phase: str | None = None,
        traders: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.1,
        strength: Annotated[float, ParameterSpec()] = 1.2,
        cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        risk: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        initial: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.traders = traders
        self.rate = rate
        self.strength = strength
        self.cost = cost
        self.risk = risk
        self.noise = noise
        self.initial = initial

    def act(self, system: SystemView) -> None:
        """Clear the price from the forecasts of the traders' heuristics and report the fitnesses.

        Args:
            system:
                The system view of the institution.
        """
        rate, strength = system.parameter('rate'), system.parameter('strength')
        state = system.storage.get('market')
        if not isinstance(state, _MarketStateSpec):
            state = _MarketStateSpec((system.parameter('initial'),))
        history = list(state.history)
        forecasts = np.array(
            [heuristics.fundamental(history), heuristics.fundamental(history, fundamental=0.0, reversion=strength)],
        )
        traders = system.agents('traders', HeuristicTraderView)
        share = float(np.mean([trader.predictor for trader in traders])) if traders else 0.0
        price = float(forecasts @ [1 - share, share] + system.rng.normal(0.0, system.parameter('noise'))) / rate
        previous = state.forecasts
        excess = price - rate * history[-1]
        if previous is None:
            fitness = np.zeros(len(PREDICTORS))
        else:
            fitness = excess * (previous - rate * history[-1]) / system.parameter('risk')
        fitness -= [system.parameter('cost'), 0.0]
        system.storage['market'] = _MarketStateSpec((history[-1], price), forecasts)
        system.report('deviation', price)
        system.report('trend_share', share)
        for name, value in zip(PREDICTORS, fitness, strict=True):
            system.report(f'fitness_{name}', value)
