"""Define topology requests, the targets of laws that add and remove the nodes and edges of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from ..base import Requires, SpaceKind, ValidationError

TOPOLOGY_KINDS = ('add-edge', 'remove-edge', 'rewire', 'add-node', 'remove-node')
EDGE_KINDS = ('add-edge', 'remove-edge', 'rewire')
ENDS = ('source', 'target')

type Toward = Callable[[Any], object] | None


@dataclass(frozen=True, eq=False)
class TopologyRequest:
    """Change of the topology of a network that a law requests for every member or edge it runs for.

    A request is the target of an `Update` law, whose value is the probability that it fires at every step for the
    member or edge the law runs for, or of a `Jump` law, as `Jump(rate, {request: 1})`, which fires it at the rate. The
    stepper applies the requests that fire after the step, in the order of the members and edges. A request on edges
    adds or removes members of the variable group on the edges of its layer, and a request on nodes adds or removes
    members of every variable group on the nodes of the network.

    Args:
        kind:
            `'add-edge'`, `'remove-edge'`, `'rewire'`, `'add-node'`, or `'remove-node'`.

        edges:
            The number of edges a new node or a node adding edges joins to other nodes.

        toward:
            A function of the namespace of a candidate node that returns its weight for a new end, such as
            `lambda node: neighbors(1)` for the degree of preferential attachment, or `None` for equal weights.

        keep:
            The end of an edge that a rewiring keeps, `'source'` or `'target'`.

        layer:
            The layer whose edges the request changes, or `None` for the graph of the network.

        values:
            The initial states of a new node or edge as pairs of a state name and a value, which default to the
            defaults of the element type.

    Raises:
        ValidationError:
            The kind or the kept end is unknown, or the number of edges is negative.

    Examples:
        >>> from skcomplex.core.laws import rewire
        >>> rewire(keep='target').kind
        'rewire'
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.NETWORK})

    kind: str
    edges: int = 1
    toward: Toward = None
    keep: str = 'source'
    layer: str | None = None
    values: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        """Check the kind, the kept end, and the number of edges."""
        if self.kind not in TOPOLOGY_KINDS:
            error_msg = f'Unknown topology request {self.kind!r}. Use one of {", ".join(TOPOLOGY_KINDS)}.'
            raise ValidationError(error_msg)
        if self.keep not in ENDS:
            error_msg = f'A rewiring keeps the source or the target of its edge, not {self.keep!r}.'
            raise ValidationError(error_msg)
        if int(self.edges) != self.edges or self.edges < 0:
            error_msg = f'A topology request adds a whole number of edges of at least 0, not {self.edges!r}.'
            raise ValidationError(error_msg)

    @property
    def on_edges(self) -> bool:
        """Whether the request changes only the edges of its layer."""
        return self.kind in EDGE_KINDS

    def __repr__(self) -> str:
        """Return the kind of the request with its layer."""
        layer = '' if self.layer is None else f', layer={self.layer!r}'
        return f'{self.kind.replace("-", "_")}({layer.removeprefix(", ")})'


def add_edge(*, toward: Toward = None, edges: int = 1, layer: str | None = None, **values: object) -> TopologyRequest:
    """Return the request that joins the node a law runs for to other nodes by new edges.

    The new ends are drawn among the other nodes present that are not yet joined to the node, with probability
    proportional to `toward`.

    Args:
        toward:
            A function of the namespace of a candidate node that returns its weight, or `None` for equal weights.

        edges:
            The number of new edges.

        layer:
            The layer of the new edges, or `None` for the graph of the network.

        **values:
            The initial states of the new edge members.

    Returns:
        request:
            The request, the target of an update or jump law of a group on the nodes.

    Examples:
        >>> from skcomplex.core.laws import add_edge
        >>> add_edge(edges=2).edges
        2
    """
    return TopologyRequest('add-edge', edges, toward, layer=layer, values=tuple(values.items()))


def remove_edge(*, layer: str | None = None) -> TopologyRequest:
    """Return the request that removes the edge a law runs for.

    Args:
        layer:
            The layer of the edge, or `None` for the graph of the network.

    Returns:
        request:
            The request, the target of an update or jump law of an interaction over neighbours.

    Examples:
        >>> from skcomplex.core.laws import remove_edge
        >>> remove_edge().kind
        'remove-edge'
    """
    return TopologyRequest('remove-edge', 0, layer=layer)


def rewire(*, keep: str = 'source', toward: Toward = None, layer: str | None = None) -> TopologyRequest:
    """Return the request that moves one end of the edge a law runs for to another node.

    The new end is drawn among the nodes present other than the kept end and not yet joined to it, with probability
    proportional to `toward`, and the edge keeps its states.

    Args:
        keep:
            The end the edge keeps, `'source'` or `'target'`.

        toward:
            A function of the namespace of a candidate node that returns its weight, or `None` for equal weights.

        layer:
            The layer of the edge, or `None` for the graph of the network.

    Returns:
        request:
            The request, the target of an update or jump law of an interaction over neighbours.

    Examples:
        >>> from skcomplex.core.laws import rewire
        >>> rewire(keep='target').keep
        'target'
    """
    return TopologyRequest('rewire', 1, toward, keep=keep, layer=layer)


def add_node(*, edges: int = 0, toward: Toward = None, layer: str | None = None, **values: object) -> TopologyRequest:
    """Return the request that adds a node to the network and joins it to other nodes by new edges.

    The new node takes a free row of every variable group on the nodes, with the given initial states or the defaults
    of the element type, and its edges join nodes drawn without repetition with probability proportional to `toward`.

    Args:
        edges:
            The number of edges of the new node.

        toward:
            A function of the namespace of a candidate node that returns its weight, or `None` for equal weights.

        layer:
            The layer of the new edges, or `None` for the graph of the network.

        **values:
            The initial states of the new node in the groups on the nodes.

    Returns:
        request:
            The request, the target of an update or jump law.

    Examples:
        >>> from skcomplex.core.laws import add_node, neighbors
        >>> add_node(edges=2, toward=lambda node: neighbors(1)).edges
        2
    """
    return TopologyRequest('add-node', edges, toward, layer=layer, values=tuple(values.items()))


def remove_node() -> TopologyRequest:
    """Return the request that removes the node a law runs for, with every edge that joins it.

    Returns:
        request:
            The request, the target of an update or jump law of a group on the nodes.

    Examples:
        >>> from skcomplex.core.laws import remove_node
        >>> remove_node().kind
        'remove-node'
    """
    return TopologyRequest('remove-node', 0)


def request_of(law: object) -> TopologyRequest | None:
    """Return the topology request an update or jump law targets, or `None` for a law that targets states.

    Args:
        law:
            The law.

    Returns:
        request:
            The request.

    Raises:
        ValidationError:
            A jump changes states and requests a topology change at once, or fires a request more than once.

    Examples:
        >>> from skcomplex.core.laws import Update, remove_node, request_of
        >>> request_of(Update(remove_node(), 0.1)).kind
        'remove-node'
    """
    target = getattr(law, 'target', None)
    if isinstance(target, TopologyRequest):
        return target
    changes = getattr(law, 'changes', None) or {}
    requests = [item for item in changes if isinstance(item, TopologyRequest)]
    if not requests:
        return None
    if len(changes) > 1 or float(changes[requests[0]]) != 1:
        error_msg = (
            f'A jump requests {requests[0]!r} next to other changes or with an increment other than 1. A jump that '
            'changes the topology fires one request, `Jump(rate, {request: 1})`.'
        )
        raise ValidationError(error_msg)
    return requests[0]
