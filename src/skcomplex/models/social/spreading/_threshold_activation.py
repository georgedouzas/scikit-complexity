"""Define threshold activation, the irreversible activation of nodes by the share of their active neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.interactions import MapInteraction
from ....core.types import ModelViewProtocol
from ....specs import RoleSpec
from ._threshold_node_view import ThresholdNodeView


class ThresholdActivationInteraction(MapInteraction):
    """Activation of the Watts cascade model, where a node becomes active once enough of its neighbours are active.

    A node becomes active for good once the share of active nodes among its neighbouring nodes reaches its threshold.
    The nodes update one after the other in a new random order every step, each reading the activations the nodes
    before it set, so a cascade can cross several nodes in one step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of nodes.

    Examples:
        >>> from skcomplex.models.social.spreading import ThresholdActivationInteraction
        >>> ThresholdActivationInteraction('activation', target='nodes').state
        'active'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='active',
            function=self._get_active,
            schedule='random-sequential',
            form='agent',
        )

    def _get_active(self, agent: ThresholdNodeView, model: ModelViewProtocol) -> float:
        """Return 1 for a node that is or becomes active, and 0 for one that stays inactive."""
        del model
        neighbors = [item for item in agent.neighbors() if isinstance(item, ThresholdNodeView)]
        if agent.active or not neighbors:
            return float(agent.active)
        return float(sum(neighbor.active for neighbor in neighbors) / len(neighbors) >= agent.threshold)
