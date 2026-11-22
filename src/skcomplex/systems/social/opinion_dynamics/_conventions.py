"""Convention systems: a shared name emerging from pairwise games."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ._random_pairs import RandomPairsInteraction


class NamerView(AgentViewProtocol, Protocol):
    """View of an agent of the naming game.

    Its states and parameters are attributes.
    """

    words: Array


class NamerElement(Element):
    """Agent of the naming game, with an inventory of the words it knows.

    The words name one object. The inventory is an indicator vector over the lexicon, with one slot per agent for the
    word that agent invents when it first speaks with an empty inventory.

    Args:
        label:
            The label of the agent or group.

        words:
            The initial inventory, 1 for every known word and 0 otherwise.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.social.opinion_dynamics import NamerElement
        >>> group = NamerElement('namers', words=np.zeros((5, 5)), size=5)
        >>> group.label, type(group).__name__, group.size
        ('namers', 'NamerElement', 5)
    """

    def __init__(
        self,
        label: str,
        *,
        words: npt.ArrayLike = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.words = words


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

    def _invent(self, words: Array, system: SystemView) -> None:
        """Mark a new word in an inventory.

        The new word is the first slot of the lexicon that no agent knows.
        """
        known = np.zeros_like(words)
        for agent in cast(list[NamerView], system.get_agents('target')):
            known += np.asarray(agent.words, dtype=np.float64)
        words[int(np.flatnonzero(known == 0)[0])] = 1.0

    def act_on(self, members: list[AgentViewProtocol], system: SystemView) -> None:
        """Play one naming game between the first member and the second.

        The first member is the speaker, and the second the hearer.

        Args:
            members:
                The speaker and the hearer.

            system:
                The system view of the rule.
        """
        speaker, hearer = cast(list[NamerView], members)
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


class NamingGameSystem(BaseSystem):
    """The minimal naming game, where agents with empty inventories agree on one word.

    The system has the group `namers`.

    Args:
        n_agents:
            The number of agents.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import NamingGameSystem
        >>> NamingGameSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'agents', 'language', 'consensus')
    references = ('Baronchelli et al. 2006', 'Castellano, Fortunato, and Loreto 2009')
    until = 50

    def __init__(
        self,
        n_agents: int = 50,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        namers = NamerElement('namers', words=np.zeros((self.n_agents, self.n_agents)), size=self.n_agents)
        self.system_ = System(
            self.label,
            elements=[namers],
            interactions=[NamingGamesInteraction('games', target='namers')],
        )
