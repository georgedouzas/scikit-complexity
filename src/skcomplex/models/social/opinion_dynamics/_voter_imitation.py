"""Define voter imitation, the copy of the opinion of a random neighbour by every voter in turn."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.interactions import MapInteraction
from ....core.types import ModelViewProtocol
from ....specs import RoleSpec
from ._voter_view import VoterView


class VoterImitationInteraction(MapInteraction):
    """Imitation of the voter model, where every voter in turn adopts the opinion of a random neighbour.

    The voters update one after the other in a new random order every step, each reading the opinions the voters before
    it set. A voter without a neighbouring voter keeps its opinion.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of voters.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import VoterImitationInteraction
        >>> VoterImitationInteraction('imitation', target='voters').schedule
        'random-sequential'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='opinion',
            function=self._get_opinion,
            schedule='random-sequential',
            form='agent',
        )

    def _get_opinion(self, agent: VoterView, model: ModelViewProtocol) -> float:
        """Return the opinion of a random neighbouring voter, or the voter's own without one."""
        neighbors = [item for item in agent.neighbors() if isinstance(item, VoterView)]
        if not neighbors:
            return float(agent.opinion)
        return float(neighbors[int(model.rng.integers(len(neighbors)))].opinion)
