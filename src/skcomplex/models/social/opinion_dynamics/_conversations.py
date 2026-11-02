"""Conversations of the cognitive consistency model, one per agent and step, the agents speaking in random order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.interactions import GroupingInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ....types import Array
from ._thinker_view import ThinkerView

TOPICS = (0, 2)


COHERENCE = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])


SETTLED = 0.01


MAX_SETTLING = 1000


def _settle_once(valences: Array, decay: float) -> Array:
    """Return the valences after one synchronous update of the parallel constraint satisfaction network."""
    net = COHERENCE @ valences
    growth = np.where(net > 0, net * (1.0 - valences), net * (valences + 1.0))
    return valences * (1.0 - decay) + growth


def settle(valences: npt.ArrayLike, decay: float = 0.05) -> Array:
    """Return the valences of a belief network updated until no valence changes by more than 0.01.

    The network links the first concept to the second with weight 0.05 and the second to the third with weight
    -0.05. Every update decays a valence by `decay` and moves it towards 1 or -1 by its net input.

    Args:
        valences:
            The valences of the three concepts, between -1 and 1.

        decay:
            The fraction of every valence lost per update.

    Returns:
        valences:
            The settled valences.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import settle
        >>> settle([1.0, 1.0, -1.0]).round(2).tolist()
        [0.48, 0.57, -0.48]
    """
    current = np.asarray(valences, dtype=float)
    for _ in range(MAX_SETTLING):
        following = _settle_once(current, decay)
        if np.abs(following - current).max() < SETTLED:
            return following
        current = following
    return current


class ConversationsInteraction(GroupingInteraction):
    """Conversations of the cognitive consistency model, one per agent and step, the agents speaking in random order.

    A speaker talks to a random neighbour about F1 or F2. The hearer averages its valence of the topic with the
    speaker's and settles its belief network, and its attitude is its valence of X.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of thinkers.

        phase:
            The phase of the step the conversations run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ConversationsInteraction
        >>> ConversationsInteraction('talks', target='thinkers').groups
        'neighborhood'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='neighborhood', phase=phase)

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Let the first member speak to a random neighbour, who averages the topic and settles its beliefs.

        Args:
            members:
                The speaker, followed by its neighbours.

            model:
                The model view of the rule.
        """
        speaker, *neighbors = [member for member in members if isinstance(member, ThinkerView)]
        if not neighbors:
            return
        hearer = neighbors[int(model.rng.integers(len(neighbors)))]
        topic = TOPICS[int(model.rng.integers(len(TOPICS)))]
        valences = np.array(hearer.valences, dtype=np.float64)
        valences[topic] = (valences[topic] + np.asarray(speaker.valences, dtype=np.float64)[topic]) / 2
        settled = settle(valences, hearer.decay)
        hearer.valences, hearer.attitude = settled, float(settled[1])
