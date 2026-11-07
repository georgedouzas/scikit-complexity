"""Markets systems: Brock and Hommes' asset market with traders switching between forecasting heuristics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ChoiceInteraction, FeedbackInteraction, ReinforcementInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import BaseSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.forecasting import heuristics


@dataclass(frozen=True)
class _MarketStateSpec(BaseSpec):
    """Last two prices and forecasts of the market, which its institution keeps in the storage of the system.

    Args:
        history:
            The last two deviations of the price, or the initial one.

        forecasts:
            The forecasts of the heuristics at the last step, or `None` before the first.
    """

    history: tuple[float, ...]
    forecasts: Array | None = None


class HeuristicTraderView(AgentView):
    """Trader as the rules see it: its heuristic `predictor` and the last fitness of every heuristic."""

    predictor: float
    fundamentalist_fitness: float
    trend_fitness: float


PREDICTORS = ('fundamentalist', 'trend')


PRICING = 'pricing'


class HeuristicTraderElement(Element):
    """Trader who holds the index of the forecasting heuristic in use, 0 for the fundamentalist and 1 for the trend.

    The trader scores every heuristic by its last realised fitness, `fundamentalist_fitness` and `trend_fitness`, which
    a reinforcement of rate 1 sets every step and a logit choice reads.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        predictor:
            The index of the heuristic in use.

        fundamentalist_fitness:
            The last realised fitness of the fundamentalist heuristic.

        trend_fitness:
            The last realised fitness of the trend heuristic.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import HeuristicTraderElement
        >>> GroupElement('traders', element=HeuristicTraderElement, size=10, values={'predictor': 1.0}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        predictor: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fundamentalist_fitness: Annotated[float, StateSpec()] = 0.0,
        trend_fitness: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.predictor = predictor
        self.fundamentalist_fitness = fundamentalist_fitness
        self.trend_fitness = trend_fitness

    @classmethod
    def _get_view(cls) -> type[HeuristicTraderView]:
        """Return the view of a speculator, with its heuristic."""
        return HeuristicTraderView

    @staticmethod
    def get_fitness(agent: HeuristicTraderView, system: SystemView, option: float) -> float:
        """Return the score of a heuristic of a trader, its last realised fitness.

        Args:
            agent:
                The trader.

            system:
                The system view of the choice.

            option:
                The index of the heuristic.

        Returns:
            fitness:
                The last realised fitness of the heuristic.
        """
        return float(getattr(agent, f'{PREDICTORS[int(option)]}_fitness'))

    @staticmethod
    def get_realised_fitness(agent: HeuristicTraderView, system: SystemView, index: int) -> float:
        """Return the fitness of a heuristic that the market maker reported last, zero before the first report.

        Args:
            agent:
                The trader.

            system:
                The system view of the reinforcement.

            index:
                The index of the heuristic.

        Returns:
            fitness:
                The last reported fitness of the heuristic.
        """
        reported = system.history(f'{PRICING}.fitness_{PREDICTORS[index]}', 1)
        return float(reported[-1]) if len(reported) else 0.0


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


class BrockHommesSystem(BaseSystem):
    """Brock and Hommes' asset market with traders switching between fundamentalists and trend chasers.

    Every step the traders score every heuristic by its last realised fitness and choose a heuristic by a logit choice
    of intensity `beta` on its last realised fitness, and the market maker clears the deviation of the price from the
    fundamental at `R x_t = sum_h n_h f_h`. The fundamental steady state is stable while the share of trend chasers
    there, `1 / (1 + exp(-beta C))`, is below `R / g`, and beyond the critical intensity `beta* = ln(R / (g - R)) / C`
    the price fluctuates persistently around the fundamental.

    The system has the group `traders`, their scores `learning` and choice `choice`, and the institution `pricing`,
    whose reporters are the deviation of the price `pricing.deviation`, the share of trend chasers
    `pricing.trend_share`, and the fitnesses `pricing.fitness_fundamentalist` and `pricing.fitness_trend`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import BrockHommesSystem
        >>> BrockHommesSystem(100).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'economics/markets'
    topics = ('asset pricing', 'heuristics', 'bounded rationality', 'agents')
    references = ('Brock and Hommes 1998',)
    until = 100

    def __init__(
        self,
        n_traders: int = 1000,
        *,
        intensity: float = 1.0,
        rate: float = 1.1,
        strength: float = 1.2,
        cost: float = 1.0,
        risk: float = 1.0,
        noise: float = 0.01,
        initial: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_traders = n_traders
        self.intensity = intensity
        self.rate = rate
        self.strength = strength
        self.cost = cost
        self.risk = risk
        self.noise = noise
        self.initial = initial

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        traders = GroupElement('traders', element=HeuristicTraderElement, size=self.n_traders)
        learning = ReinforcementInteraction(
            'learning',
            target='traders',
            scores=tuple(f'{name}_fitness' for name in PREDICTORS),
            payoff=HeuristicTraderElement.get_realised_fitness,
            rate=1.0,
            form='agent',
            phase='learning',
        )
        choice = ChoiceInteraction(
            'choice',
            target='traders',
            state='predictor',
            options=(0, 1),
            utility=HeuristicTraderElement.get_fitness,
            intensity=self.intensity,
            form='agent',
            phase='choice',
        )
        pricing = HeuristicPricingInteraction(
            PRICING,
            traders='traders',
            phase='pricing',
            rate=self.rate,
            strength=self.strength,
            cost=self.cost,
            risk=self.risk,
            noise=self.noise,
            initial=self.initial,
        )
        return System(
            self.label,
            elements=[traders],
            interactions=[learning, choice, pricing],
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec('learning'), PhaseSpec('choice'), PhaseSpec('pricing')]),
            ),
        )
