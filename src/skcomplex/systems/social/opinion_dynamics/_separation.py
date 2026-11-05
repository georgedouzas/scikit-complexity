"""Rewiring of discordant edges by one end, which keeps the edge and moves its other end to a like-minded node."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import TopologyInteraction
from ....core.specs import RoleSpec

SMALLEST_DEGREE = 1e-9


class SeparationInteraction(TopologyInteraction):
    """Rewiring of discordant edges by one end, which keeps the edge and moves its other end to a like-minded node.

    It is a `TopologyInteraction` that rewires the edges of the voters. The end `keep` acts on an edge with
    probability `activity / degree` per step, and when it holds the opinion `opinion` and the other end does not, it
    rewires the edge with probability `rewiring`, drawing a new end among the voters of the opinion `opinion`.

    Args:
        label:
            The label of the interaction.

        voters:
            The path of the voters on the nodes.

        keep:
            The end of an edge that acts and keeps the edge, `'source'` or `'target'`.

        opinion:
            The opinion of the end that acts, 0 or 1.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import SeparationInteraction
        >>> SeparationInteraction('separation', voters='voters', keep='target', opinion=0).action
        'rewire'
    """

    def __init__(
        self,
        label: str,
        *,
        voters: Annotated[str, RoleSpec()],
        keep: str = 'source',
        opinion: float = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=voters,
            action='rewire',
            probability=self._get_chance,
            toward=self._get_weight,
            keep=keep,
        )
        self.voters = voters
        self.keep = keep
        self.opinion = opinion

    def _get_weight(self, node: SimpleNamespace) -> object:
        """Return the weight of a candidate node as a new end, 1 when it holds the opinion of the acting end."""
        return node.opinion if self.opinion == 1 else 1 - node.opinion

    def _get_chance(self, namespace: SimpleNamespace) -> object:
        """Return the probability that the acting end rewires the edge, when the ends hold different opinions."""
        actor, other = (namespace.source, namespace.target)[:: 1 if self.keep == 'source' else -1]
        chance = actor.activity * actor.rewiring / (actor.degree + SMALLEST_DEGREE)
        if self.opinion == 1:
            return chance * actor.opinion * (1 - other.opinion)
        return chance * (1 - actor.opinion) * other.opinion
