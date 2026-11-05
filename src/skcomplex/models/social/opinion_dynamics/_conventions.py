"""Convention models: a shared name emerging from pairwise games."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.system import System
from ..._catalog import register_model
from ._namer import NamerElement
from ._naming_games import NamingGamesInteraction


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'language', 'consensus'),
    references=('Baronchelli et al. 2006', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 50},
)
def naming_game(n_agents: int = 50) -> System:
    """Return the minimal naming game, where agents with empty inventories agree on one word.

    Args:
        n_agents:
            The number of agents.

    Returns:
        system:
            The model, with the group `namers`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import naming_game
        >>> naming_game(10).elements[0].prepare().size_
        10
    """
    namers = GroupElement(
        'namers',
        element=NamerElement,
        size=n_agents,
        values={'words': np.zeros((n_agents, n_agents))},
    )
    return System(elements=[namers], interactions=[NamingGamesInteraction('games', target='namers')])
