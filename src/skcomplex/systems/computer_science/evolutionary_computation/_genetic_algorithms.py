"""Evolutionary computation systems: a genetic algorithm that evolves Robby the can-collecting robot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import EvolutionInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ._room_cleaning import RoomCleaningInteraction
from ._strategy import StrategyElement

N_SITUATIONS = 243


N_ACTIONS = 7


@register_system(
    field='computer_science/evolutionary_computation',
    tags=('genetic algorithm', 'agents', 'evolution', 'robot'),
    references=('Mitchell 2009', 'Gooding 2019'),
    defaults={'until': 20},
)
def robby_ga(
    n_strategies: int = 50,
    *,
    mutation: float = 0.005,
    sessions: int = 10,
    n_actions: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Mitchell's genetic algorithm that evolves strategies for Robby, a robot that collects cans in a room.

    Each step is one generation. Parents are selected with probabilities proportional to their rank, every pair of
    parents exchanges the tails of their strategies after a random cut, and every action of a child mutates to another
    action with probability `mutation`. Then every child is scored in random rooms. Random strategies score far below
    zero because they bump into walls, and evolution finds strategies that avoid walls and collect cans.

    Args:
        n_strategies:
            The size of the population.

        mutation:
            The probability that an action of a child is replaced by another one.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        random_state:
            The seed or generator of the random first generation and of its scores.

    Returns:
        system:
            The system, with the group `strategies` of states `actions` and `score`.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import robby_ga
        >>> robby_ga(5).elements[0].values['actions'].shape
        (5, 243)
    """
    rng = np.random.default_rng(random_state)
    evolution = EvolutionInteraction(
        'evolution',
        target='strategies',
        strategy='actions',
        fitness='score',
        selection='rank',
        crossover=1.0,
        mutation=mutation,
        alphabet=N_ACTIONS,
    )
    scoring = RoomCleaningInteraction('scoring', target='strategies', sessions=sessions, n_actions=n_actions)
    actions = rng.integers(0, N_ACTIONS, size=(n_strategies, N_SITUATIONS))
    scores = scoring.score(actions, rng)
    return System(
        elements=[
            GroupElement(
                'strategies',
                element=StrategyElement,
                size=n_strategies,
                values={'actions': actions, 'score': scores},
            ),
        ],
        interactions=[evolution, scoring],
    )
