"""Econophysics systems: the minority game and its evolutionary version."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import ChoiceInteraction, EvolutionInteraction, ReinforcementInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ....core.specs import PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_system
from ._minority_game import MinorityGameInteraction
from ._minority_player import MinorityPlayerElement

MAX_SEED = 2**31


@register_system(
    field='economics/econophysics',
    tags=('minority game', 'agents', 'inductive reasoning', 'evolution'),
    references=('Challet and Zhang 1997', 'Yakovenko and Rosser 2009', 'Carmichael and Hadzikadic 2019 ch. 9'),
    defaults={'until': 500},
)
def minority_game(
    n_agents: int = 101,
    *,
    memory: int = 3,
    evolutionary: bool = False,
    period: int = 50,
    fraction: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the minority game of an odd number of agents with two random strategies each.

    Args:
        n_agents:
            The number of agents, odd so that there is always a minority.

        memory:
            The memory `m` of the strategies, so `alpha = 2^m / N`.

        evolutionary:
            Whether every `period` rounds the poorest agents take new random strategies.

        period:
            The number of rounds between two evolutions of the evolutionary version.

        fraction:
            The fraction of poorest agents that take new strategies at an evolution.

        random_state:
            The seed or generator of the strategies.

    Returns:
        system:
            The system, with the group `players`, the choice `choice` of the strategy of higher virtual score, the
            interaction `game`, which reports the attendance `game.attendance`, the cumulative reinforcement
            `scoring` of the virtual scores, and in the evolutionary version the truncation `evolution`, which gives
            the poorest agents new random strategies and no wealth every `period` rounds.

    Raises:
        ValueError:
            The number of agents is even.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import minority_game
        >>> minority_game(11, memory=2).interactions[1].memory
        2
    """
    if n_agents % 2 == 0:
        error_msg = f'The minority game needs an odd number of agents, got {n_agents}.'
        raise ValueError(error_msg)
    codes = np.random.default_rng(random_state).integers(MAX_SEED, size=(2, n_agents)).astype(float)
    players = GroupElement(
        'players',
        element=MinorityPlayerElement,
        size=n_agents,
        values={'first': codes[0], 'second': codes[1]},
    )
    choice = ChoiceInteraction(
        'choice',
        target='players',
        state='strategy',
        options=(0.0, 1.0),
        utility=lambda agent, system, option: agent.second_score if option else agent.first_score,
        rule='best',
        form='agent',
        phase='choose',
    )
    game = MinorityGameInteraction('game', players='players', memory=memory, phase='play')
    scoring = ReinforcementInteraction(
        'scoring',
        target='players',
        scores=('first_score', 'second_score'),
        payoff=game.payoff,
        rule='cumulative',
        form='agent',
        phase='score',
    )
    evolution = EvolutionInteraction(
        'evolution',
        target='players',
        strategy=('first', 'second'),
        fitness='wealth',
        replacement='truncation',
        fraction=fraction,
        mutation=1.0,
        change='any',
        alphabet=MAX_SEED,
        period=period,
        reset={'first_score': 0.0, 'second_score': 0.0, 'wealth': 0.0},
        phase='score',
    )
    return System(
        elements=[players],
        interactions=[choice, game, scoring, *([evolution] if evolutionary else [])],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('choose'), PhaseSpec('play'), PhaseSpec('score')]),
        ),
    )
