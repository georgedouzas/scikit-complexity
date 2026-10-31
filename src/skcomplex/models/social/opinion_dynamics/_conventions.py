"""Convention models: a shared name emerging from pairwise games."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.elements import Element, GroupElement
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._namer_view import NamerView
from ._random_pairs import RandomPairsInteraction


class NamerElement(Element):
    """Agent of the naming game, with an inventory of the words it knows for one object.

    The inventory is an indicator vector over the lexicon, with one slot per agent for the word that agent invents
    when it first speaks with an empty inventory.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        words:
            The initial inventory, 1 for every known word and 0 otherwise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.social.opinion_dynamics import NamerElement
        >>> group = GroupElement('namers', element=NamerElement, size=5, values={'words': np.zeros((5, 5))})
        >>> group.label, group.element.__name__, group.size
        ('namers', 'NamerElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        words: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.words = words

    @classmethod
    def _get_view(cls) -> type[NamerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return NamerView


class NamingGamesInteraction(RandomPairsInteraction):
    """Pairwise games of the minimal naming game, as many per step as there are agents.

    A random speaker says a random word of its inventory to a random hearer. When the hearer knows it, both keep that
    word alone, and otherwise the hearer adds it. A speaker with an empty inventory first invents a word that no agent
    knows.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of namers.

        phase:
            The phase of the step the games run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import NamingGamesInteraction
        >>> NamingGamesInteraction('games', target='namers').target
        'namers'
    """

    def _invent(self, words: Array, model: ModelViewProtocol) -> None:
        """Mark in an inventory a new word, the first slot of the lexicon that no agent knows."""
        known = np.zeros_like(words)
        for agent in model.agents('target'):
            if isinstance(agent, NamerView):
                known += np.asarray(agent.words, dtype=np.float64)
        words[int(np.flatnonzero(known == 0)[0])] = 1.0

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Play one naming game between the first member, the speaker, and the second, the hearer.

        Args:
            members:
                The speaker and the hearer.

            model:
                The model view of the rule.
        """
        speaker, hearer = [member for member in members if isinstance(member, NamerView)]
        spoken = np.array(speaker.words, dtype=np.float64)
        heard = np.array(hearer.words, dtype=np.float64)
        if not spoken.any():
            self._invent(spoken, model)
        known = np.flatnonzero(spoken)
        name = known[int(model.rng.integers(len(known)))]
        if heard[name]:
            spoken[:] = heard[:] = 0.0
            spoken[name] = heard[name] = 1.0
        else:
            heard[name] = 1.0
        speaker.words, hearer.words = spoken, heard


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
        >>> naming_game(10).elements[0].check().prepare().size_
        10
    """
    namers = GroupElement(
        'namers',
        element=NamerElement,
        size=n_agents,
        values={'words': np.zeros((n_agents, n_agents))},
    )
    return System(elements=[namers], interactions=[NamingGamesInteraction('games', target='namers')])
