"""Define the layer spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import networkx as nx

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class LayerSpec(BaseSpec):
    """Spec of a relation over the nodes of a network space, such as contacts or employment, held as a graph of its own.

    A network space holds several layers over the same nodes. Edge groups and interactions over neighbours name the
    layer they bind to, and reductions reduce over its edges with `AdjacencyNeighborhood(layer=label)`.

    Args:
        label:
            The name of the layer, unique in its network space.

        graph:
            The NetworkX graph of the relation, over the nodes of the network space.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.specs import LayerSpec
        >>> LayerSpec('work', nx.path_graph(3)).check().label
        'work'
    """

    label: str
    graph: nx.Graph

    def _check(self) -> None:
        """Raise an error for a label that is not a string.

        Raises:
            ValidationError:
                The label is not a string.
        """
        if not isinstance(self.label, str):
            error_msg = f'The label of a layer must be a string, but it is {self.label!r}.'
            raise ValidationError(error_msg)
