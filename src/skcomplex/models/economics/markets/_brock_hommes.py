"""Markets models: Brock and Hommes' asset market with traders switching between forecasting heuristics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.elements import GroupElement
from ....core.interactions import ChoiceInteraction, ReinforcementInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._heuristic_pricing import HeuristicPricingInteraction
from ._heuristic_trader import PREDICTORS, PRICING, HeuristicTraderElement


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

    Every step the traders score every heuristic by its last realised fitness and choose a heuristic by a logit choice
    of intensity `beta` on its last realised fitness, and the market maker clears the deviation of the price from the
    fundamental at `R x_t = sum_h n_h f_h`. The fundamental steady state is stable while the share of trend chasers
    there, `1 / (1 + exp(-beta C))`, is below `R / g`, and beyond the critical intensity `beta* = ln(R / (g - R)) / C`
    the price fluctuates persistently around the fundamental.

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
            The model, with the group `traders`, their scores `learning` and choice `choice`, and the institution
            `pricing`, whose reporters are the deviation of the price `pricing.deviation`, the share of trend chasers
            `pricing.trend_share`, and the fitnesses `pricing.fitness_fundamentalist` and `pricing.fitness_trend`.

    Examples:
        >>> from skcomplex.models.economics.markets import brock_hommes
        >>> brock_hommes(100).elements[0].prepare().size_
        100
    """
    traders = GroupElement('traders', element=HeuristicTraderElement, size=n_traders)
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
        intensity=intensity,
        form='agent',
        phase='choice',
    )
    pricing = HeuristicPricingInteraction(
        PRICING,
        traders='traders',
        phase='pricing',
        rate=rate,
        strength=strength,
        cost=cost,
        risk=risk,
        noise=noise,
        initial=initial,
    )
    return System(
        elements=[traders],
        interactions=[learning, choice, pricing],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('learning'), PhaseSpec('choice'), PhaseSpec('pricing')]),
        ),
    )
