"""Define host graph elements, graphs rewritten by double-pushout graph productions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, Self

import networkx as nx

from ..errors import RewritingError
from ..specs import StateSpec, VisualSpec
from ._element import Element


class HostGraphElement(Element):
    """Element that holds a graph rewritten by double-pushout graph productions.

    A subclass returns the `GraphProductionLaw`s of the graph from `_get_laws`. Its states are the count of nodes of
    every `label` attribute that is a word, and the number `size` of nodes.

    Args:
        label:
            The label of the element.

        graph:
            The graph at generation 0, directed or not, or `None` for the empty graph.

        max_size:
            The largest number of nodes a generation may produce.

        max_matches:
            The largest number of matches searched per production and generation.

        visual:
            How the graph is drawn, or `None` for `default_visual_`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.elements import HostGraphElement
        >>> HostGraphElement('ring', graph=nx.cycle_graph(4)).check().graph.number_of_nodes()
        4
    """

    def __init__(
        self,
        label: str,
        graph: Annotated[nx.Graph | None, StateSpec(domain=None)] = None,
        *,
        max_size: int = 1_000_000,
        max_matches: int = 100_000,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.graph = graph
        self.max_size = max_size
        self.max_matches = max_matches

    def _check(self) -> Self:
        """Raise an error for a graph that is not a NetworkX graph.

        Returns:
            element:
                The element itself.
        """
        if self.graph is not None and not isinstance(self.graph, nx.Graph):
            error_msg = f'The graph of {self.label} must be a NetworkX graph, but it is {type(self.graph).__name__}.'
            raise RewritingError(error_msg)
        return self

    def _prepare(self, context: object = None) -> Self:
        """Set `default_visual_`, the geometry of the graph.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.

        Returns:
            element:
                The element itself.
        """
        self.default_visual_ = VisualSpec('geometry')
        return self
