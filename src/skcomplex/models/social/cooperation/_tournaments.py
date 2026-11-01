"""Tournament models: the evolution of strategies of the iterated prisoner's dilemma."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import EvolutionInteraction
from ....core.system import System
from ....specs import VisualSpec
from ..._catalog import register_model
from ._strategist import StrategistElement
from ._tournament import TournamentInteraction

GENES = 7


@register_model(
    field='social/cooperation',
    tags=('agents', 'games', 'cooperation', 'evolution'),
    references=('Axelrod 1984', 'Downey 2018'),
    defaults={'until': 50},
)
def prisoners_dilemma_tournament(
    n_agents: int = 100,
    *,
    rounds: int = 6,
    death: float = 0.1,
    mutation: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the evolution of strategies of the iterated prisoner's dilemma from random genomes.

    Args:
        n_agents:
            The number of agents.

        rounds:
            The number of rounds of every game.

        death:
            The probability that an agent dies every step.

        mutation:
            The probability that an offspring has one gene flipped.

        random_state:
            The seed or generator of the initial genomes.

    Returns:
        system:
            The model, with the group `agents`, the tournament `tournament`, and the death-birth replacement
            `selection`.

    Examples:
        >>> from skcomplex.models.social.cooperation import prisoners_dilemma_tournament
        >>> prisoners_dilemma_tournament(10, random_state=0).elements[0].check().prepare().size_
        10
    """
    genomes = np.random.default_rng(random_state).integers(0, 2, (n_agents, GENES)).astype(float)
    visual = VisualSpec('histograms', {'color_by': 'fitness'})
    agents = GroupElement(
        'agents',
        element=StrategistElement,
        size=n_agents,
        values={'genome': genomes},
        visual=visual,
    )
    return System(
        elements=[agents],
        interactions=[
            TournamentInteraction('tournament', target='agents', rounds=rounds),
            EvolutionInteraction(
                'selection',
                target='agents',
                strategy='genome',
                selection='proportional',
                replacement='death-birth',
                mutation=mutation,
                death=death,
                change='one',
                parents='all',
            ),
        ],
    )
