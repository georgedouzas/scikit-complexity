"""Spatial games: cooperation on a lattice where players imitate more successful neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.elements import LatticeElement
from ....core.interactions import ImitationInteraction
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....specs import PhaseSpec, PhasesScheduleSpec
from ....types import Array
from ..._catalog import register_model
from ._contributor import ContributorElement
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


@register_model(
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
            The model, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.models.social.cooperation import spatial_prisoners_dilemma
        >>> spatial_prisoners_dilemma((10, 10)).elements[0].check().prepare().size_
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


@register_model(
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
            The model, with the strategies `players.strategy` and the payoffs `players.payoff`.

    Examples:
        >>> from skcomplex.models.social.cooperation import public_goods_game
        >>> public_goods_game((10, 10)).elements[0].check().prepare().size_
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
