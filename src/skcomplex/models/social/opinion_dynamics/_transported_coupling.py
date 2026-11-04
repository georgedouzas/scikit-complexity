"""Coupling of agent phases through the connection on the edge that joins them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sin

from ....core.interactions import CouplingInteraction
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, RoleSpec


class TransportedCouplingInteraction(CouplingInteraction):
    """Relaxation of agent phases towards the phases of their neighbours, `theta_i' += K sin(theta_j - theta_i - A_ij)`.

    The connection `A_ij` lives on the edge of the network from `i` to `j`, and `A_ji = -A_ij`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of agents on the nodes.

        links:
            The path of the group on the edges, whose parameter `A` holds the connection.

        K:
            The coupling strength.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import TransportedCouplingInteraction
        >>> TransportedCouplingInteraction('coupling', target='agents', links='links', K=0.5).K
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, target=target, read='theta', write='theta')
        self.links = links
        self.K = K

    def _get_edge_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of an edge, with the connection read along and against its direction."""
        source, target = namespace.source, namespace.target
        flow = namespace.K * sin(target.theta - source.theta - namespace.edge.A)
        return [RateLaw(source.theta, flow), RateLaw(target.theta, -flow)]
