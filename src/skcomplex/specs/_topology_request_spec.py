"""Define the topology request spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from ..errors import ValidationError
from ._base_spec import BaseSpec

TOPOLOGY_KINDS = ('add-edge', 'remove-edge', 'rewire', 'add-node', 'remove-node')
EDGE_KINDS = ('add-edge', 'remove-edge', 'rewire')
ENDS = ('source', 'target')
MIN_HYPEREDGE = 2


@dataclass(frozen=True, eq=False)
class TopologyRequestSpec(BaseSpec):
    """Spec of a change of the topology of a network that a law requests for every member or edge it runs for.

    A request is the target of an `UpdateLaw`, whose value is the probability that it fires at every step for the
    member or edge the law runs for, or of a `JumpLaw`, as `JumpLaw(rate, {request: 1})`, which fires it at the rate.
    The stepper applies the requests that fire after the step, in the order of the members and edges. A request on
    edges, `'add-edge'`, `'remove-edge'`, or `'rewire'`, adds or removes members of the variable group on the edges of
    its layer, and a request on nodes, `'add-node'` or `'remove-node'`, adds or removes members of every variable group
    on the nodes of the network. New ends are drawn among the nodes present that are not yet joined to the node, with
    probability proportional to `toward`.

    Args:
        kind:
            `'add-edge'`, `'remove-edge'`, `'rewire'`, `'add-node'`, or `'remove-node'`.

        edges:
            The number of edges a new node or a node adding edges joins to other nodes.

        toward:
            A function of the namespace of a candidate node that returns its weight for a new end, such as
            `lambda node: neighbors(1)` for the degree of preferential attachment, or `None` for equal weights. The
            namespace holds the node the new edge starts from as `node.source`.

        keep:
            The end of an edge that a rewiring keeps, `'source'` or `'target'`.

        layer:
            The layer whose edges the request changes, or `None` for the graph of the network.

        values:
            The initial states of a new node or edge as pairs of a state name and a value, or a function of the
            random number generator that draws the value for every new node or edge, which default to the defaults of
            the element type.

        hyperedge:
            The size of the hyperedge a new node forms with existing nodes, joining each of them by an edge in place
            of `edges`, or `None` for no hyperedge. The first two existing members are the ends of an edge drawn
            uniformly, and further members are drawn by `toward`.

    Examples:
        >>> from skcomplex.core.laws import UpdateLaw
        >>> from skcomplex.specs import TopologyRequestSpec
        >>> UpdateLaw(TopologyRequestSpec('rewire', keep='target').prepare(), 0.1).target.keep
        'target'
    """

    kind: str
    edges: int = 1
    toward: Callable[[SimpleNamespace], object] | None = None
    keep: str = 'source'
    layer: str | None = None
    values: tuple[tuple[str, object], ...] = ()
    hyperedge: int | None = None

    def _check(self) -> None:
        """Raise an error for an unknown kind or kept end, or a number of edges that is not a whole number.

        Raises:
            ValidationError:
                The kind or the kept end is unknown, the number of edges is negative or not whole, or a hyperedge
                has fewer than two members or belongs to a request other than `'add-node'`.
        """
        if self.kind not in TOPOLOGY_KINDS:
            error_msg = f'Unknown topology request {self.kind!r}. Use one of {", ".join(TOPOLOGY_KINDS)}.'
            raise ValidationError(error_msg)
        if self.keep not in ENDS:
            error_msg = f'A rewiring keeps the source or the target of its edge, not {self.keep!r}.'
            raise ValidationError(error_msg)
        if int(self.edges) != self.edges or self.edges < 0:
            error_msg = f'A topology request adds a whole number of edges of at least 0, not {self.edges!r}.'
            raise ValidationError(error_msg)
        if self.hyperedge is None:
            return
        if int(self.hyperedge) != self.hyperedge or self.hyperedge < MIN_HYPEREDGE:
            error_msg = f'A new node forms a hyperedge of a whole number of at least 2 members, not {self.hyperedge!r}.'
            raise ValidationError(error_msg)
        if self.kind != 'add-node':
            error_msg = f'A {self.kind!r} request forms no hyperedge. Only a request that adds a node takes one.'
            raise ValidationError(error_msg)
