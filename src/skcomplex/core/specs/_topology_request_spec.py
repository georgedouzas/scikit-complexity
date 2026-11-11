"""Define the topology request spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from ..enums import TopologyKind
from ..errors import ValidationError
from ._base_spec import BaseSpec

EDGE_KINDS = frozenset(
    {
        TopologyKind.ADD_EDGE,
        TopologyKind.REMOVE_EDGE,
        TopologyKind.REWIRE,
        TopologyKind.CLOSE_TRIANGLE,
        TopologyKind.CONTRACT_EDGE,
        TopologyKind.DUPLICATE_NODE,
    },
)
HYPEREDGE_KINDS = frozenset({TopologyKind.ADD_NODE, TopologyKind.ADD_HYPEREDGE})
ENDS = ('source', 'target')
DIRECTIONS = ('out', 'in', 'both')
MIN_HYPEREDGE = 2


@dataclass(frozen=True, eq=False)
class TopologyRequestSpec(BaseSpec):
    """A spec of a change of the topology of a network that a law requests.

    The law requests it for every member or edge it runs for. A request is the target of an `UpdateLaw`, whose value is
    the probability that it fires at every step for the member or edge the law runs for, or of a `JumpLaw`, as
    `JumpLaw(rate, {request: 1})`, which fires it at the rate. The stepper applies the requests that fire after the
    step, in the order of the members and edges. A request on edges, `'add-edge'`, `'remove-edge'`, or `'rewire'`, adds
    or removes members of the variable group on the edges of its layer, and a request on nodes, `'add-node'` or
    `'remove-node'`, adds or removes members of every variable group on the nodes of the network. A request on
    hyperedges adds or removes members of the variable group on the hyperedges: `'add-hyperedge'` runs for a node, which
    forms a new hyperedge with `hyperedge - 1` other nodes drawn by `toward`, and `'remove-hyperedge'` runs for every
    hyperedge present and removes it. New ends are drawn among the nodes present that are not yet joined to the node,
    with probability proportional to `toward`, or with `independent` every such node becomes an end on its own with the
    probability `toward`, so the number of new ends is random.

    Further requests run for a node. `'replace-node'` removes it and adds a new node that joins new ends like
    `'add-node'`. `'close-triangle'` joins two of its neighbours that are not yet joined, drawn uniformly among such
    pairs, or, for a node without such a pair, joins it to `edges` nodes like `'add-edge'`.
    `'contract-edge'` draws a neighbour by `toward` and merges it into the node, which keeps the union of their links
    on every layer, while the neighbour leaves. `'duplicate-node'` adds a copy of the node with its values, which
    inherits each of its links with probability `inherit`, joins it with probability `parent`, and joins new ends
    among the nodes joined to neither like a new node. An `'add-edge'` with `candidates` draws that many candidate ends
    for every new edge and adds the one of the smallest `score`. Without a target it runs once per step, and every
    candidate joins two distinct nodes not yet joined, drawn uniformly. With `one`, a request runs at every step for
    one member drawn uniformly among the members present, in place of every member.

    Args:
        kind:
            `'add-edge'`, `'remove-edge'`, `'rewire'`, `'add-node'`, `'remove-node'`, `'add-hyperedge'`,
            `'remove-hyperedge'`, `'replace-node'`, `'close-triangle'`, `'contract-edge'`, or `'duplicate-node'`.

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
            The size of the hyperedge a new node or the node of `'add-hyperedge'` forms with existing nodes, or `None`
            for no hyperedge. A new node with at least one edge to draw and no `independent` ends joins each member by
            an edge in place of `edges`, and its first two existing members are the ends of an edge drawn uniformly.
            Otherwise every other member is drawn by `toward`, and the hyperedge adds no edge.

        independent:
            Whether every candidate node becomes a new end on its own with the probability `toward`, clipped to
            `[0, 1]`, in place of `edges` ends drawn by weight.

        direction:
            The direction of the new edges of a node on a directed layer: `'out'` from the node, `'in'` into it, or
            `'both'` for an edge each way, drawn apart with `independent` ends.

        candidates:
            The number of candidate edges an `'add-edge'` request draws for every edge it adds.

        score:
            A function of the namespace of a candidate end that returns the score of its edge, which reads the other
            end as `node.source`, such as the product of the sizes of the components the edge joins, or `None` for
            the first candidate. The candidate of the smallest score is added, so a negated score adds the largest.

        one:
            Whether the request runs at every step for one member drawn uniformly among the members present, in place
            of every member.

        inherit:
            The probability that the copy of `'duplicate-node'` inherits each link of the node.

        parent:
            The probability that the copy of `'duplicate-node'` joins the node.

    Examples:
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> from skcomplex.core.specs import TopologyRequestSpec
        >>> UpdateLaw(TopologyRequestSpec('rewire', keep='target'), 0.1).target.keep
        'target'
    """

    kind: str
    edges: int = 1
    toward: Callable[[SimpleNamespace], object] | None = None
    keep: str = 'source'
    layer: str | None = None
    values: tuple[tuple[str, object], ...] = ()
    hyperedge: int | None = None
    independent: bool = False
    direction: str = 'out'
    candidates: int = 1
    score: Callable[[SimpleNamespace], object] | None = None
    one: bool = False
    inherit: float = 1.0
    parent: float = 0.0

    def _check(self) -> None:
        """Raise an error for an unknown kind or kept end, or a bad number of edges.

        The number of edges must be a whole number.

        Raises:
            ValidationError:
                The kind, the kept end, or the direction is unknown, the number of edges is negative or not whole, a
                hyperedge has fewer than two members or belongs to a request other than `'add-node'` and
                `'add-hyperedge'`, `'add-hyperedge'` has no hyperedge, the candidates are not a whole number of at
                least 1 or belong to a request other than `'add-edge'` or to independent ends, a score is not
                callable or belongs to a request other than `'add-edge'`, or a probability of duplication lies outside
                `[0, 1]`.
        """
        if self.kind not in TopologyKind:
            error_msg = f'Unknown topology request {self.kind!r}. Use one of {", ".join(TopologyKind)}.'
            raise ValidationError(error_msg)
        if self.keep not in ENDS:
            error_msg = f'A rewiring keeps the source or the target of its edge, not {self.keep!r}.'
            raise ValidationError(error_msg)
        if self.direction not in DIRECTIONS:
            error_msg = f'New edges leave a node, enter it, or both, {", ".join(DIRECTIONS)}, not {self.direction!r}.'
            raise ValidationError(error_msg)
        if int(self.edges) != self.edges or self.edges < 0:
            error_msg = f'A topology request adds a whole number of edges of at least 0, not {self.edges!r}.'
            raise ValidationError(error_msg)
        self._check_choice()
        self._check_duplication()
        if self.hyperedge is None:
            if self.kind == 'add-hyperedge':
                error_msg = "An 'add-hyperedge' request forms a hyperedge of a size. Give it `hyperedge`."
                raise ValidationError(error_msg)
            return
        if int(self.hyperedge) != self.hyperedge or self.hyperedge < MIN_HYPEREDGE:
            error_msg = f'A node forms a hyperedge of a whole number of at least 2 members, not {self.hyperedge!r}.'
            raise ValidationError(error_msg)
        if self.kind not in HYPEREDGE_KINDS:
            error_msg = (
                f'A {self.kind!r} request forms no hyperedge. Only a request that adds a node or a hyperedge takes one.'
            )
            raise ValidationError(error_msg)

    def _check_choice(self) -> None:
        """Raise an error for candidates or a score outside a choice among new edges.

        They must belong to a choice among new edges.

        Raises:
            ValidationError:
                The candidates are not a whole number of at least 1, or more than one belong to a request other than
                `'add-edge'` or to independent ends, or the score is not callable or belongs to another request.
        """
        if int(self.candidates) != self.candidates or self.candidates < 1:
            error_msg = f'A request draws a whole number of at least 1 candidates, not {self.candidates!r}.'
            raise ValidationError(error_msg)
        chooses = self.candidates > 1 or self.score is not None
        if chooses and (self.kind != 'add-edge' or self.independent):
            error_msg = (
                f'A {self.kind!r} request chooses among candidates. Only an \'add-edge\' request without independent '
                'ends takes candidates and a score.'
            )
            raise ValidationError(error_msg)
        if self.score is not None and not callable(self.score):
            error_msg = f'The score of a request is {self.score!r}. Give a function of the namespace, or None.'
            raise ValidationError(error_msg)

    def _check_duplication(self) -> None:
        """Raise an error for a probability of duplication outside `[0, 1]`.

        Raises:
            ValidationError:
                The probability that a copy inherits a link or joins its original lies outside `[0, 1]`.
        """
        for name in ('inherit', 'parent'):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                error_msg = f'The duplication probability {name} is {value!r}. Give a number in [0, 1].'
                raise ValidationError(error_msg)
