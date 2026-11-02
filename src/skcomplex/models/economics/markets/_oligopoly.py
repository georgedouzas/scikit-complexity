"""Static oligopoly equilibria, and the fictitious play of a discretised pricing duopoly."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import ChoiceInteraction, ReinforcementInteraction
from ....core.spaces import DiscreteTime, StaticTime
from ....core.system import System
from ....errors import ValidationError
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ....types import Array
from ..._catalog import register_model
from ._cournot_best_response import CournotBestResponseElement
from ._price_point import PricePointElement
from ._pricing_firm import PricingFirmElement
from ._pricing_game_spec import PricingGameSpec

DYNAMICS = ('static', 'fictitious-play')
DUOPOLY = 2


def _price_observables(system: System) -> list[ObservableSpec]:
    """Return the lowest and the highest expected profit over the prices, or the prices charged under learning."""
    if isinstance(system.time, StaticTime):
        profit = PricePointElement.get_profit
        return [
            ObservableSpec('lowest_profit', profit, group='prices', reduce='min'),
            ObservableSpec('highest_profit', profit, group='prices', reduce='max'),
        ]
    return [
        ObservableSpec('lower_price', group='firms', variable='price', reduce='min'),
        ObservableSpec('higher_price', group='firms', variable='price', reduce='max'),
    ]


def _fictitious_play(prices: Array, alpha: float, beta: float) -> System:
    """Return two firms that best respond to the sum of the payoffs of every price against the rival so far."""
    game = PricingGameSpec(prices, alpha, beta)
    firms = GroupElement(
        'firms',
        element=PricingFirmElement,
        size=DUOPOLY,
        values={'price': np.full(DUOPOLY, prices[-1]), 'scores': np.zeros((DUOPOLY, prices.size))},
    )
    choice = ChoiceInteraction(
        'choice',
        target='firms',
        state='price',
        options=tuple(prices.tolist()),
        utility=game.utility,
        rule='best',
        form='agent',
        phase='choose',
    )
    beliefs = ReinforcementInteraction(
        'beliefs',
        target='firms',
        scores='scores',
        payoff=game.payoff,
        rule='cumulative',
        form='agent',
        phase='learn',
    )
    schedule = PhasesScheduleSpec([PhaseSpec('choose'), PhaseSpec('learn')])
    return System(elements=[firms], interactions=[choice, beliefs], time=DiscreteTime(schedule=schedule))


@register_model(
    field='economics/markets',
    tags=('oligopoly', 'equilibrium', 'static', 'game theory'),
    references=('Forrest 2018', 'Brown 1951'),
    observables=_price_observables,
)
def oligopoly_equilibria(
    n_firms: int = 2,
    *,
    alpha: float = 0.25,
    n_prices: int = 21,
    demand: float = 10.0,
    c_i: float = 1.0,
    c_j: float = 2.0,
    lambda_i: float = 0.0,
    lambda_j: float = 0.0,
    dynamics: str = 'static',
) -> System:
    """Return the equilibria of a Cournot duopoly and of a pricing game of firms with loyal customers and switchers.

    The static model solves the Cournot best responses (Forrest 2018 Eqs. 21.5-21.6) and, on a grid of prices, the
    symmetric mixed equilibrium of `m` firms of loyal share `alpha` and switchers `beta = 1 - m alpha`, whose price
    distribution makes a firm indifferent among its prices, at the profit `alpha` (Lemma 20.1). With
    `dynamics='fictitious-play'`, two firms learn to price on the grid instead: every round each plays a best response
    to the prices of its rival so far, and the prices they charge are distributed near the mixed equilibrium.

    Args:
        n_firms:
            The number of firms of the pricing game, two under fictitious play.

        alpha:
            The loyal share of every firm.

        n_prices:
            The number of prices of the grid from `alpha / (alpha + beta)` to 1.

        demand:
            The intercept of the inverse demand of the Cournot duopoly.

        c_i:
            The unit cost of the first Cournot firm.

        c_j:
            The unit cost of the second Cournot firm.

        lambda_i:
            The cross holding of the first Cournot firm.

        lambda_j:
            The cross holding of the second Cournot firm.

        dynamics:
            `'static'` for the equilibria, or `'fictitious-play'` for the learning of two pricing firms.

    Returns:
        system:
            The static model, with the states `cournot.q_i`, `cournot.q_j`, `prices.price`, and `prices.distribution`
            and the observables `lowest_profit` and `highest_profit`, or the learning model, with the group `firms` and
            the observables `lower_price` and `higher_price`. The catalog lists the observables, so a run passes them
            in `observables`.

    Raises:
        ValidationError:
            The dynamics are unknown, or fictitious play has other than two firms.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.economics.markets import oligopoly_equilibria
        >>> trajectory = Simulation(oligopoly_equilibria()).check().prepare().run(1.0).trajectory_
        >>> round(float(trajectory.values('cournot.q_i')[0]), 6)
        3.333333
    """
    if dynamics not in DYNAMICS or (dynamics == 'fictitious-play' and n_firms != DUOPOLY):
        error_msg = (
            f'oligopoly_equilibria has dynamics={dynamics!r} and n_firms={n_firms!r}. Use one of {DYNAMICS}, and '
            f'{DUOPOLY} firms under fictitious play.'
        )
        raise ValidationError(error_msg)
    beta = 1 - n_firms * alpha
    prices = np.linspace(alpha / (alpha + beta), 1.0, n_prices)
    if dynamics == 'fictitious-play':
        return _fictitious_play(prices, alpha, beta)
    points = GroupElement(
        'prices',
        element=PricePointElement,
        size=n_prices,
        values={'price': prices, 'alpha': alpha, 'beta': beta, 'n_firms': float(n_firms)},
    )
    cournot = CournotBestResponseElement(
        'cournot',
        demand=demand,
        c_i=c_i,
        c_j=c_j,
        lambda_i=lambda_i,
        lambda_j=lambda_j,
    )
    return System(elements=[cournot, points], time=StaticTime())
