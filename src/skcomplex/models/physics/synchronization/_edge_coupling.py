"""Kuramoto coupling along the weighted edges of a network, `theta_i' += K w_ij sin(theta_j - theta_i)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....specs import RoleSpec
from ..common.oscillators import KuramotoCouplingInteraction


class EdgeCouplingInteraction(KuramotoCouplingInteraction):
    """Kuramoto coupling along the weighted edges of a network, `theta_i' += K w_ij sin(theta_j - theta_i)`.

    The coupling runs once per edge of the network of its oscillators and reads the weight `w` of the edge from the
    group on the edges.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators on the nodes.

        links:
            The path of the group on the edges, whose parameter `w` weighs every edge.

        K:
            The coupling strength.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.models.physics.synchronization import EdgeCouplingInteraction
        >>> EdgeCouplingInteraction('coupling', target='nodes', links='edges', K=0.5).weights
        'w'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: float = 1.0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(label, target=target, K=K, alpha=alpha, weights='w')
        self.links = links
