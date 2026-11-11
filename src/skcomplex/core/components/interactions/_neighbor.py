"""Define the interaction whose laws run once for every edge of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ..base import BaseComponent
from ._base import BaseInteraction


class NeighborInteraction(BaseInteraction):
    """An interaction whose laws run once for every edge of a network or lattice.

    A role binds a group with one member per node or site, and a second role may bind a group on the edges of the same
    network, such as the conductance of each line of a power grid. The laws are written for one edge. They read
    `namespace.source` and `namespace.target`, the members at its two ends, `namespace.edge`, the member of the edge
    group, and the parameters. On a directed graph the ends follow the direction of the edge, and on an undirected graph
    the source comes first in node order. On fixed edges `namespace.source.degree` and `namespace.target.degree` are the
    numbers of edges at each end.

    A law that reads no edge member and only one end outside its reductions runs once per member, not once per edge.
    Inside a reduction the other end stands for the other members, so `mean(f(s.source.x, s.target.x), over=s.source)`
    is the mean of `f` over the neighbours of each target. A second role may also bind a second group of nodes, with no
    edges, and the reductions then run over its members. Roles may bind separate elements too, and `namespace.source` is
    then the tuple of their namespaces. A subclass that names its ends otherwise returns their names from `_get_role`.

    Unlike `Interaction`, which runs once per bound member, it runs once per edge, so each law sees both ends of a link.
    Choose it when a change flows along the links of a lattice or network, such as diffusion or a flow on a line.

    It appears in many fields. Heat flows between neighbouring cells of a lattice, a disease spreads along the contacts
    of a network, and power flows along the lines of a grid.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'diffusion'`.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the layer of the edge group it
            binds, or else the graph of the network.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.interactions import NeighborInteraction
        >>> from skcomplex.core.specs import ParameterSpec, RoleSpec
        >>> class BondExchangeInteraction(NeighborInteraction):
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
        >>> BondExchangeInteraction('diffusion', nodes='cells', layer='roads').layer
        'roads'
    """

    def __init__(self, label: str, layer: str | None = None) -> None:
        super().__init__(label)
        self.layer = layer

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the laws of the interaction for one edge, none by default.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return []
