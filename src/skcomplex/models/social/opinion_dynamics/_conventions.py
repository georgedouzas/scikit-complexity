"""Convention models: a shared name emerging from pairwise games."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.system import System

type Array = npt.NDArray[np.float64]


def _play(speaker: AgentView, hearer: AgentView, own: int, rng: np.random.Generator) -> None:
    """Play one naming game between a speaker and a hearer, the speaker inventing its own word when it knows none."""
    spoken = np.array(speaker.words, dtype=float)
    heard = np.array(hearer.words, dtype=float)
    if not spoken.any():
        spoken[own] = 1.0
    known = np.flatnonzero(spoken)
    name = known[int(rng.integers(len(known)))]
    if heard[name]:
        spoken[:] = heard[:] = 0.0
        spoken[name] = heard[name] = 1.0
    else:
        heard[name] = 1.0
    speaker.words, hearer.words = spoken, heard


class Namer(Element):
    """Agent of the naming game, with an inventory of the words it knows for one object.

    The inventory is an indicator vector over the lexicon, with one slot per agent for the word that agent invents
    when it first speaks with an empty inventory.

    Args:
        label:
            The label of the agent or group.

        words:
            The initial inventory, 1 for every known word and 0 otherwise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.social.opinion_dynamics import Namer
        >>> Namer.group('namers', 5, words=np.zeros((5, 5)))
        Group('namers', element=Namer, size=5)
    """

    words: npt.ArrayLike = state(default=0.0, vector=True)


class NamingGames(Interaction):
    """Pairwise games of the minimal naming game, as many per step as there are agents.

    A random speaker says a random word of its inventory to a random hearer. When the hearer knows it, both keep that
    word alone, and otherwise the hearer adds it.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of namers.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import NamingGames
        >>> NamingGames('games', target='namers').target
        'namers'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the game rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.play)]

    def play(self, model: ModelView) -> None:
        """Play one game per agent between random pairs.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if len(agents) < 2:  # noqa: PLR2004
            return
        for _ in range(len(agents)):
            speaker, hearer = model.rng.choice(len(agents), size=2, replace=False)
            _play(agents[speaker], agents[hearer], int(speaker), model.rng)


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
        >>> naming_game(10).elements[0].size
        10
    """
    namers = Namer.group('namers', n_agents, words=np.zeros((n_agents, n_agents)))
    return System(components=[namers], interactions=[NamingGames('games', target='namers')])
