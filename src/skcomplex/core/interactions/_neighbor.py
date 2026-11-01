"""Define neighbour interactions, whose laws run for every edge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ..base import BasePart
from ._base import BaseInteraction


class NeighborInteraction(BaseInteraction):
    """Base class of interactions whose laws run once for every edge of the network of the nodes a role binds.

    A role binds a group with one member per node or site, and a second role may bind a group on the edges of the same
    network and layer. The laws are written for one edge and read `namespace.source` and `namespace.target`, the
    members at both ends, `namespace.edge`, the member of the edge group, and the parameters. On a directed graph the
    ends follow the direction of the edge, and on an undirected graph the source comes first in node order. Rate,
    noise, and jump laws may target any of them, and update laws the edge member. `delayed` reads the past state of
    either end.

    On fixed edges, `namespace.source.degree` and `namespace.target.degree` are the degrees of the ends, the number of
    edges of the layer at every node, both ways on a directed graph. A law that reads no edge member and, outside its
    reductions, only one end is a law of that end's member, and runs once per member rather than once per edge. Inside
    its reductions the other end stands for the other members of the group, so
    `mean(f(namespace.source.x, namespace.target.x), over=namespace.source)` is the mean of `f` over every member for
    the target. Cascade, drive, and local rule laws are laws of the members, and a cascade records its cascades
    as events.

    Instead of a group on the edges, a second role may bind a second group of nodes. The laws then run once per member
    of the group of the first role, which `namespace.target` reads, and their reductions run over the members of the
    second group, which `namespace.source` reads, without edges. A role declared `RoleSpec(optional=True)` and bound to
    `None` binds nothing.

    The roles may also bind separate elements. When the first role binds one element or a list of elements, the laws
    run once per element, which `namespace.target` reads. When the second role binds a group, `namespace.source` reads
    its members inside reductions, and otherwise `namespace.source` is the tuple of the namespaces of the elements the
    second role binds, or of the other elements of the list the first role binds.

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

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the laws of the interaction for one edge, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
