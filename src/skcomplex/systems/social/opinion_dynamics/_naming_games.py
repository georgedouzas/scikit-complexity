"""Pairwise games of the minimal naming game, as many per step as there are agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.types import Array
from ._namer_view import NamerView
from ._random_pairs import RandomPairsInteraction


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
        >>> from skcomplex.systems.social.opinion_dynamics import NamingGamesInteraction
        >>> NamingGamesInteraction('games', target='namers').target
        'namers'
    """

    def _invent(self, words: Array, system: SystemViewProtocol) -> None:
        """Mark in an inventory a new word, the first slot of the lexicon that no agent knows."""
        known = np.zeros_like(words)
        for agent in system.agents('target'):
            if isinstance(agent, NamerView):
                known += np.asarray(agent.words, dtype=np.float64)
        words[int(np.flatnonzero(known == 0)[0])] = 1.0

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Play one naming game between the first member, the speaker, and the second, the hearer.

        Args:
            members:
                The speaker and the hearer.

            system:
                The system view of the rule.
        """
        speaker, hearer = [member for member in members if isinstance(member, NamerView)]
        spoken = np.array(speaker.words, dtype=np.float64)
        heard = np.array(hearer.words, dtype=np.float64)
        if not spoken.any():
            self._invent(spoken, system)
        known = np.flatnonzero(spoken)
        name = known[int(system.rng.integers(len(known)))]
        if heard[name]:
            spoken[:] = heard[:] = 0.0
            spoken[name] = heard[name] = 1.0
        else:
            heard[name] = 1.0
        speaker.words, hearer.words = spoken, heard
