"""Spatial games: cooperation on a lattice where players imitate more successful neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.components.elements import LatticeElement
from ....core.components.interactions import ImitationInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.specs import PhaseSpec, PhasesScheduleSpec
from ....core.types import Array
from ..._catalog import register_system
from ._contributor import ContributorElement
from ._iterated_dilemma import IteratedDilemmaInteraction
from ._iterated_prisoner import IteratedPrisonerElement
from ._prisoner import PrisonerElement
from ._prisoners_dilemma import PrisonersDilemmaInteraction
from ._public_goods import PublicGoodsInteraction


def _random_strategies(
    shape: Sequence[int],
    share: float,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return strategies that are 1, cooperation, with a probability and 0, defection, otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < share).astype(float)


@register_system(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'evolution'),
    references=('Nowak and May 1992', 'Szabó and Fáth 2007', 'Argun et al. 2021', 'Gros 2015'),
    defaults={'until': 50, 'dt': 1},
)
def spatial_prisoners_dilemma(
    shape: Sequence[int] = (50, 50),
    *,
    temptation: float = 1.85,
    imitation: str = 'best',
    noise: float = 0.1,
    share: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial prisoner's dilemma of Nowak and May, with best-neighbour or Fermi imitation.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        temptation:
            The payoff of a defector against a cooperator.

        imitation:
            `'best'` to copy the best neighbour, or `'fermi'` to copy a random neighbour with the Fermi probability.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The system, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import spatial_prisoners_dilemma
        >>> spatial_prisoners_dilemma((10, 10)).elements[0].prepare().size_
        100
    """
    players = LatticeElement(
        'players',
        element=PrisonerElement,
        space=LatticeSpace(tuple(shape)),
        values={
            'strategy': _random_strategies(shape, share, random_state),
            'temptation': temptation,
            'noise': noise,
        },
    )
    return System(
        elements=[players],
        interactions=[PrisonersDilemmaInteraction('play', target='players', imitation=imitation)],
    )


@register_system(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'public goods'),
    references=('Szabó and Hauert 2002', 'Szabó and Fáth 2007'),
    defaults={'until': 50, 'dt': 1},
)
def public_goods_game(
    shape: Sequence[int] = (50, 50),
    *,
    synergy: float = 4.5,
    noise: float = 0.5,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial public goods game in groups of five neighbours, with Fermi imitation.

    Every step has two phases. In the phase `play` every player takes the payoff of its five groups, and in the phase
    `imitate` every player copies a random neighbour with the Fermi probability `1 / (1 + exp((P_own - P_other) / K))`
    of the payoffs of the same step, at the noise `K`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        synergy:
            The factor that multiplies the contributions of a group.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The system, with the strategies `players.strategy` and the payoffs `players.payoff`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import public_goods_game
        >>> public_goods_game((10, 10)).elements[0].prepare().size_
        100
    """
    strategies = _random_strategies(shape, share, random_state)
    players = LatticeElement(
        'players',
        element=ContributorElement,
        space=LatticeSpace(tuple(shape), neighborhood='von_neumann'),
        values={'strategy': strategies, 'synergy': synergy, 'noise': noise},
    )
    imitation = ImitationInteraction(
        'imitation',
        target='players',
        state='strategy',
        rule='fermi',
        form='probability',
        intensity=1 / noise,
        phase='imitate',
    )
    time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
    return System(
        elements=[players],
        interactions=[PublicGoodsInteraction('play', target='players', phase='play'), imitation],
        time=time,
    )


@register_system(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'iterated', 'mutation'),
    references=('Argun et al. 2021', 'Axelrod and Hamilton 1981'),
    defaults={'until': 100, 'dt': 1},
)
def iterated_lattice_dilemma(
    shape: Sequence[int] = (30, 30),
    *,
    rounds: int = 7,
    reward: float = 0.6,
    sucker: float = 1.5,
    temptation: float = 0.0,
    punishment: float = 1.0,
    mutation: float = 0.01,
    extremes: bool = False,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Argun's evolutionary iterated prisoner's dilemma on a lattice, with strategies of `n` cooperation rounds.

    The payoffs are years in prison, `T < R < P < S`, so lower is better. A strategy `n` cooperates until round `n`
    or until the opponent defects, and defects afterwards, so `n = 0` always defects and `n = N` always cooperates.
    Against a strategy `m` the best reply is `n = m - 1`. Every step each player plays the `N`-round game with its four
    von Neumann neighbours, adopts the strategy of the lowest total among itself and its neighbours, random on ties,
    and mutates to a random strategy with the probability `mu` (Argun et al. 2021 sections 13.2 and 13.3). With every
    strategy allowed and `S = 1.5`, cooperative strategies prevail for `R` below about 0.5, all strategies coexist and
    cycle between about 0.5 and 0.75, and defecting strategies dominate above, so the default `R = 0.6` lies in the
    middle of the cycling regime, where a finite lattice is not absorbed by defection. With only the two extreme
    strategies,
    `extremes=True`, cooperation dominates at `R = 0.82`, coexists with defection at `R = 0.84`, and vanishes at
    `R = 0.86` (Exercise 13.3).

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        rounds:
            The number `N` of rounds of a game.

        reward:
            The punishment `R` of mutual cooperation.

        sucker:
            The punishment `S` of a betrayed cooperator.

        temptation:
            The punishment `T` of a defector who betrays.

        punishment:
            The punishment `P` of mutual defection.

        mutation:
            The probability `mu` of a mutation to a random strategy per player and step.

        extremes:
            Whether only the strategies 0, always defect, and `N`, always cooperate, are allowed.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The system, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import iterated_lattice_dilemma
        >>> iterated_lattice_dilemma((10, 10)).elements[0].prepare().size_
        100
    """
    rng = np.random.default_rng(random_state)
    if extremes:
        strategies = rounds * rng.integers(2, size=tuple(shape)).astype(float)
    else:
        strategies = rng.integers(rounds + 1, size=tuple(shape)).astype(float)
    players = LatticeElement(
        'players',
        element=IteratedPrisonerElement,
        space=LatticeSpace(tuple(shape), neighborhood='von_neumann'),
        values={
            'strategy': strategies,
            'rounds': rounds,
            'reward': reward,
            'sucker': sucker,
            'temptation': temptation,
            'punishment': punishment,
            'mutation': mutation,
            'extremes': float(extremes),
        },
    )
    return System(elements=[players], interactions=[IteratedDilemmaInteraction('play', target='players')])
