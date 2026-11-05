"""Define variant voting, the update of imitators, threshold voters, and contrarians in turn."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.interactions import MapInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import RoleSpec
from ._variant_voter_view import VariantVoterView


class VariantVotingInteraction(MapInteraction):
    """Voting of the voter system variants, where every voter in turn takes the opinion of its variant.

    A voter of threshold 0 copies a random neighbour. A voter of positive threshold adopts the opinion opposite to its
    own when the share of neighbours that hold it exceeds the threshold. A contrarian adopts the opposite of what it
    would otherwise adopt. The voters update one after the other in a new random order every step, each reading the
    opinions the voters before it set, and a voter without a neighbouring voter keeps its opinion.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of voters.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VariantVotingInteraction
        >>> VariantVotingInteraction('voting', target='voters').schedule
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

    def _get_choice(
        self,
        agent: VariantVoterView,
        neighbors: list[VariantVoterView],
        system: SystemViewProtocol,
    ) -> float:
        """Return the opinion a voter that conforms would take."""
        if agent.threshold <= 0:
            return float(neighbors[int(system.rng.integers(len(neighbors)))].opinion)
        disagreeing = sum(neighbor.opinion != agent.opinion for neighbor in neighbors) / len(neighbors)
        return float(1 - agent.opinion) if disagreeing > agent.threshold else float(agent.opinion)

    def _get_opinion(self, agent: VariantVoterView, system: SystemViewProtocol) -> float:
        """Return the opinion of the voter's variant, inverted for a contrarian."""
        neighbors = [item for item in agent.neighbors() if isinstance(item, VariantVoterView)]
        if not neighbors:
            return float(agent.opinion)
        choice = self._get_choice(agent, neighbors, system)
        return 1 - choice if agent.contrarian else choice
