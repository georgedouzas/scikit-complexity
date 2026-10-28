"""Define neighbour interactions, whose laws run for every edge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction


class NeighborInteraction(BaseInteraction):
    """Base class of interactions whose laws run once for every edge of the network of the nodes a role binds.

    A role binds a group with one member per node or site, and a second role may bind a group on the edges of the same
    network and layer. The laws are written for one edge and read `namespace.source` and `namespace.target`, the
    members at both ends, `namespace.edge`, the member of the edge group, and the parameters. On a directed graph the
    ends follow the direction of the edge, and on an undirected graph the source comes first in node order. Rate,
    noise, and jump laws may target any of them, and update laws the edge member.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the layer of the edge group it
            binds, or else the graph of the network.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.interactions import NeighborInteraction
        >>> from skcomplex.specs import ParameterSpec, RoleSpec
        >>> class DiffusionInteraction(NeighborInteraction):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         nodes: Annotated[str, RoleSpec()],
        ...         D: Annotated[float, ParameterSpec()] = 1.0,
        ...         layer=None,
        ...     ):
        ...         super().__init__(label, layer=layer)
        ...         self.nodes = nodes
        ...         self.D = D
        >>> DiffusionInteraction('diffusion', nodes='cells', layer='roads').layer
        'roads'
    """

    def __init__(self, label: str, layer: str | None = None) -> None:
        super().__init__(label)
        self.layer = layer

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the laws of the interaction for one edge, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
