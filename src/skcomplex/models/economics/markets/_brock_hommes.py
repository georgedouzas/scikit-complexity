"""Markets models: Brock and Hommes' asset market with traders switching between forecasting heuristics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import (
    AgentType,
    Institution,
    Law,
    Phase,
    Rule,
    Symbols,
    System,
    heuristics,
    logit_choice,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

PREDICTORS = ('fundamentalist', 'trend')


class Speculator(AgentType):
    """Trader who chooses a forecasting heuristic by a logit choice on the last realised fitness of every heuristic.

    The heuristic is 0 for the fundamentalist and 1 for the trend chaser, and the fitnesses are the reporters of the
    pricing institution, zero before its first report.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        predictor:
            The index of the heuristic in use.

        intensity:
            The intensity of choice `beta`.

        pricing:
            The label of the pricing institution whose fitness reporters the trader reads.

    Examples:
        >>> from skcomplex.models.economics.markets import Speculator
        >>> Speculator.group('traders', 10, intensity=2.0).size
        10
    """

    pricing: ClassVar[str] = 'pricing'
    predictor: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    intensity: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the choice of heuristic in the phase of choice.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.choose, phase='choice')]

    def choose(self, agent: AgentView, model: ModelView) -> None:
        """Choose a heuristic with logit probabilities of the last realised fitnesses.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        agent.predictor = float(logit_choice(self._fitness(model), agent.intensity, model.rng))

    def _fitness(self, model: ModelView) -> list[float]:
        """Return the last fitness of every heuristic, read once per step into the storage of the traders."""
        time, fitness = model.storage.get('fitness', (None, []))
        if time != model.time:
            values = [model.history(f'{self.pricing}.fitness_{name}', 1) for name in PREDICTORS]
            fitness = [float(value[-1]) if len(value) else 0.0 for value in values]
            model.storage['fitness'] = (model.time, fitness)
        return fitness


class HeuristicPricing(Institution):
    """Market maker that clears the risky asset at the price where the mean forecast of the traders is met.

    The deviation `x` of the price from its fundamental value clears at `R x_t = sum_h n_h f_h + eps_t`, where `n_h`
    is the share of traders using heuristic `h`. The fundamentalists forecast the fundamental, `f = 0`, and the trend
    chasers extrapolate the last deviation, `f = g x_{t-1}`, both as `heuristics.fundamental` with a reversion of 0
    and `g`. The realised fitness of a heuristic is the profit of its last forecast,
    `(x_t - R x_{t-1}) (f_h(t - 1) - R x_{t-1}) / (a sigma^2)`, minus its cost.

    Args:
        label:
            The label of the institution.

        traders:
            The path of the group of traders.

        rule:
            Unused, since the institution overrides `act`.

        phase:
            The phase the institution clears in.

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
        >>> from skcomplex.models.economics.markets import HeuristicPricing
        >>> HeuristicPricing('pricing', traders='traders').reports
        ('deviation', 'trend_share', 'fitness_fundamentalist', 'fitness_trend')
    """

    reports: ClassVar[tuple[str, ...]] = ('deviation', 'trend_share', 'fitness_fundamentalist', 'fitness_trend')
    traders: str = role()
    rate: float = parameter(default=1.1, domain=Domain.POSITIVE)
    strength: float = parameter(default=1.2, domain=Domain.REAL)
    cost: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    risk: float = parameter(default=1.0, domain=Domain.POSITIVE)
    noise: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    initial: float = parameter(default=0.1, domain=Domain.REAL)

    def act(self, model: Any) -> None:  # noqa: ANN401
        """Clear the price from the forecasts of the traders' heuristics and report the fitnesses.

        Args:
            model:
                The model view of the institution.
        """
        rate, strength = model.parameter('rate'), model.parameter('strength')
        history = model.storage.setdefault('history', [model.parameter('initial')])
        forecasts = np.array(
            [heuristics.fundamental(history), heuristics.fundamental(history, fundamental=0.0, reversion=strength)],
        )
        traders = model.agents('traders')
        share = float(np.mean([trader.predictor for trader in traders])) if traders else 0.0
        price = (forecasts @ [1 - share, share] + model.rng.normal(0.0, model.parameter('noise'))) / rate
        previous = model.storage.get('forecasts')
        excess = price - rate * history[-1]
        if previous is None:
            fitness = np.zeros(len(PREDICTORS))
        else:
            fitness = excess * (previous - rate * history[-1]) / model.parameter('risk')
        fitness -= [model.parameter('cost'), 0.0]
        model.storage['forecasts'] = forecasts
        history[:] = [history[-1], price]
        model.report('deviation', price)
        model.report('trend_share', share)
        for name, value in zip(PREDICTORS, fitness, strict=True):
            model.report(f'fitness_{name}', value)


@register_model(
    field='economics/markets',
    tags=('asset pricing', 'heuristics', 'bounded rationality', 'agents'),
    references=('Brock and Hommes 1998',),
    defaults={'until': 100},
)
def brock_hommes(
    n_traders: int = 1000,
    *,
    intensity: float = 1.0,
    rate: float = 1.1,
    strength: float = 1.2,
    cost: float = 1.0,
    risk: float = 1.0,
    noise: float = 0.01,
    initial: float = 0.1,
) -> System:
    """Return Brock and Hommes' asset market with traders switching between fundamentalists and trend chasers.

    Every step the traders choose a heuristic by a logit choice of intensity `beta` on its last realised fitness, and
    the market maker clears the deviation of the price from the fundamental at `R x_t = sum_h n_h f_h`. The
    fundamental steady state is stable while the share of trend chasers there, `1 / (1 + exp(-beta C))`, is below
    `R / g`, and beyond the critical intensity `beta* = ln(R / (g - R)) / C` the price fluctuates persistently around
    the fundamental.

    Args:
        n_traders:
            The number of traders.

        intensity:
            The intensity of choice `beta`.

        rate:
            The gross return `R` of the riskless asset.

        strength:
            The extrapolation `g` of the trend chasers.

        cost:
            The cost `C` of the fundamentalist heuristic.

        risk:
            The risk aversion times the variance of returns, `a sigma^2`.

        noise:
            The standard deviation of the noise of the price.

        initial:
            The initial deviation of the price from its fundamental value.

    Returns:
        system:
            The model, with the group `traders` and the institution `pricing`, whose reporters are the deviation of the
            price `pricing.deviation`, the share of trend chasers `pricing.trend_share`, and the fitnesses
            `pricing.fitness_fundamentalist` and `pricing.fitness_trend`.

    Examples:
        >>> from skcomplex.models.economics.markets import brock_hommes
        >>> brock_hommes(100).elements[0].size
        100
    """
    traders = Speculator.group('traders', n_traders, intensity=intensity)
    pricing = HeuristicPricing(
        'pricing',
        traders='traders',
        phase='pricing',
        rate=rate,
        strength=strength,
        cost=cost,
        risk=risk,
        noise=noise,
        initial=initial,
    )
    return System(elements=[traders], interactions=[pricing], phases=[Phase('choice'), Phase('pricing')])
