"""Define the interaction that changes the nodes and edges of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from ...errors import ValidationError
from ...specs import BaseSpec, RoleSpec, Tags, TopologyRequestSpec
from ...types import ElementKind
from ..laws import BaseLaw, JumpLaw, UpdateLaw
from ..laws.utils import where
from ..spaces import BaseSpace
from ._neighbor import NeighborInteraction

ACTIONS = (
    'add-node',
    'remove-node',
    'replace-node',
    'add-edge',
    'remove-edge',
    'rewire',
    'add-hyperedge',
    'remove-hyperedge',
    'close-triangle',
    'contract-edge',
    'duplicate-node',
)


type Chance = float | Callable[[SimpleNamespace], object]


class TopologyInteraction(NeighborInteraction):
    """An interaction that adds, removes, and rewires the nodes and edges of a network.

    The `action` runs for every node of the group the target binds, or for every edge of its network, and fires at a
    `rate`, with a `probability` per step, or always, where `condition` holds. The groups on the nodes and edges of the
    network are variable. The actions are:

    - `'add-node'`: a new node joins `edges` other nodes. Without a target it adds one node from outside per step, as in
      a growing network.
    - `'remove-node'` and `'replace-node'`: the node leaves, or leaves and a new node takes its place in one request.
    - `'add-edge'`: the node joins `edges` other nodes, or with `candidates` the candidate of the smallest `score`.
    - `'remove-edge'` and `'rewire'`: the edge leaves, or keeps its end `keep` and moves its other end.
    - `'add-hyperedge'` and `'remove-hyperedge'`: a node forms a hyperedge of size `hyperedge`, or a hyperedge leaves.
    - `'close-triangle'`: the node joins two of its neighbours that are not yet joined.
    - `'contract-edge'`: a neighbour merges into the node, which keeps the union of their links.
    - `'duplicate-node'`: a copy of the node inherits each of its links with probability `inherit`.

    The rate, the probability, and the condition are numbers or functions of the namespace of an edge, whose ends are
    `namespace.source` and `namespace.target`, and an action on a node reads the node as the source. New ends are drawn
    with probability proportional to `toward`, or uniformly. A new node or edge takes `values`.

    Unlike `NeighborInteraction`, which runs on fixed edges, it changes the edges and nodes themselves during a run.
    Choose it when the network grows, shrinks, or rewires as the system runs, so structure and states change together.

    It appears in many fields. Preferential attachment grows the web and citation networks, people rewire their contacts
    to avoid infection, genes duplicate in the evolution of protein networks, and friends of friends close triangles in
    social networks.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'attachment'`.

        target:
            The path of the group on the nodes of the network, of the variable group on its hyperedges for
            `'remove-hyperedge'`, or `None` for an addition of one node or of edges between drawn nodes per step.

        action:
            The change, `'add-node'`, `'remove-node'`, `'replace-node'`, `'add-edge'`, `'remove-edge'`, `'rewire'`,
            `'add-hyperedge'`, `'remove-hyperedge'`, `'close-triangle'`, `'contract-edge'`, or `'duplicate-node'`.

        rate:
            The rate of the change, a number or a function, or `None` for a probability per step.

        probability:
            The probability of the change per step, a number or a function, or `None` for probability one.

        condition:
            A function of the namespace that returns where the change may fire, or `None` everywhere.

        toward:
            A function of the namespace of a candidate node that returns its weight as a new end, reading the node
            the new edge starts from as `namespace.source`, or `None` for equal weights.

        edges:
            The number of edges a new node or a node adding edges joins.

        keep:
            The end of an edge a rewiring keeps, `'source'` or `'target'`.

        values:
            The initial states of a new node or edge by name, each a value or a function of the random number
            generator that draws it, which default to the defaults of the element type.

        layer:
            The layer of the network whose edges change, or `None` for the graph of the network.

        hyperedge:
            The size of the hyperedge every new node or every node of `'add-hyperedge'` forms with existing nodes, or
            `None` for none.

        independent:
            Whether every candidate becomes a new end on its own with the probability `toward`, in place of `edges`
            ends drawn by weight.

        direction:
            The direction of new edges on a directed layer, `'out'` of the node, `'in'` to it, or `'both'`.

        candidates:
            The number of candidate edges of `'add-edge'` for every edge it adds.

        score:
            A function of the namespace of a candidate end that returns the score of its edge, reading the other end
            as `namespace.source`, whose smallest value is added, or `None` for the first candidate.

        one:
            Whether the action runs at every step for one member drawn uniformly among those present.

        inherit:
            The probability that the copy of `'duplicate-node'` inherits each link of the node.

        parent:
            The probability that the copy of `'duplicate-node'` joins the node.

    Raises:
        ValidationError:
            The action, the kept end, or the direction is unknown, both a rate and a probability are given, the
            number of edges is not a whole number, an action other than `'add-node'` and `'add-edge'` has no target, a
            hyperedge is given to an action other than `'add-node'` and `'add-hyperedge'` or has fewer than two
            members, `'add-hyperedge'` has no hyperedge, candidates or a score belong to another action than
            `'add-edge'`, or a probability of duplication lies outside `[0, 1]`, raised by `prepare`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.interactions import TopologyInteraction
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=20)
        >>> links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
        >>> expiry = TopologyInteraction('expiry', target='nodes', action='remove-edge')
        >>> system = CompositeSystem(
        ...     elements=[nodes, links], interactions=[expiry], space=NetworkSpace(nx.cycle_graph(5))
        ... )
        >>> trajectory = simulate(system, until=1, random_state=0)
        >>> len(trajectory.get_network(0).sources), len(trajectory.get_network(1).sources)
        (5, 0)

        A network that grows by triangles from one triangle:

        >>> import xgi
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.engine.runtime import simulate
        >>> triangles = GroupElement('triangles', element=Element, on='hyperedges', variable=True, capacity=8)
        >>> growth = TopologyInteraction('growth', action='add-node', hyperedge=3)
        >>> space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
        >>> system = CompositeSystem(
        ...     elements=[nodes, links, triangles], interactions=[growth], space=space, time=DiscreteTime()
        ... )
        >>> trajectory = simulate(system, until=5, random_state=0)
        >>> trajectory.get_network(-1).hyperedges.shape
        (6, 3)
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (JumpLaw, UpdateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of jumps at a rate or of updates with a chance.

        These change the topology.
        """
        changing = frozenset({'changing topology'})
        if self._get_argument('rate') is not None:
            return self._get_variant_tags((JumpLaw,), structure=changing)
        drawn = self._get_argument('probability') is not None
        return self._get_variant_tags((UpdateLaw,), stochastic=drawn, structure=changing)

    def _get_draws(self) -> bool:
        """Return whether the change draws.

        It draws by an action that draws new ends or that runs for one member.
        """
        removal = self._get_argument('action') in ('remove-node', 'remove-edge', 'remove-hyperedge')
        return super()._get_draws() or bool(self._get_argument('one')) or not removal

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str | None, RoleSpec(optional=True, elements=(ElementKind.ANY,))] = None,
        action: str = 'add-edge',
        rate: Chance | None = None,
        probability: Chance | None = None,
        condition: Callable[[SimpleNamespace], object] | None = None,
        toward: Callable[[SimpleNamespace], object] | None = None,
        edges: int = 1,
        keep: str = 'source',
        values: Mapping[str, object] | None = None,
        layer: str | None = None,
        hyperedge: int | None = None,
        independent: bool = False,
        direction: str = 'out',
        candidates: int = 1,
        score: Callable[[SimpleNamespace], object] | None = None,
        one: bool = False,
        inherit: float = 1.0,
        parent: float = 0.0,
    ) -> None:
        super().__init__(label, layer=layer)
        self.target = target
        self.action = action
        self.rate = rate
        self.probability = probability
        self.condition = condition
        self.toward = toward
        self.edges = edges
        self.keep = keep
        self.values = values
        self.hyperedge = hyperedge
        self.independent = independent
        self.direction = direction
        self.candidates = candidates
        self.score = score
        self.one = one
        self.inherit = inherit
        self.parent = parent

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for unknown options, a rate with a probability, or bad edges.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The action or the kept end is unknown, both a rate and a probability are given, a condition, a
                weight, or a score cannot be called, the number of edges is not a whole number of at least 0, an
                action other than `'add-node'` and `'add-edge'` has no target, a hyperedge is given to an action other
                than `'add-node'` and `'add-hyperedge'`, or `'add-hyperedge'` has none, or the request is invalid.
        """
        super()._prepare(context)
        self._check_variant('action', self.action, ACTIONS, (BaseSpec,))
        if self.rate is not None and self.probability is not None:
            error_msg = f'{self.label!r} fires at a rate and with a probability. Give one of them.'
            raise ValidationError(error_msg)
        if self.target is None and self.action not in {'add-node', 'add-edge'}:
            error_msg = f'{self.label!r} runs {self.action!r} without a target. Give the group on the nodes.'
            raise ValidationError(error_msg)
        if self.hyperedge is not None and self.action not in {'add-node', 'add-hyperedge'}:
            error_msg = (
                f'{self.label!r} runs {self.action!r} with a hyperedge. Only new nodes and new hyperedges take a size.'
            )
            raise ValidationError(error_msg)
        for name in ('condition', 'toward', 'score'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace, or None.'
                raise ValidationError(error_msg)
        for request in self._get_requests():
            request._check()

    def _get_requests(self) -> list[TopologyRequestSpec]:
        """Return the request of the action."""
        if self.action not in ACTIONS:
            raise self._get_variant_error(self.action, 'action', '_get_requests')
        return [
            TopologyRequestSpec(
                self.action,
                self.edges,
                self.toward,
                self.keep,
                self.layer,
                tuple((self.values or {}).items()),
                self.hyperedge if self.action in {'add-node', 'add-hyperedge'} else None,
                self.independent,
                self.direction,
                self.candidates,
                self.score,
                self.one,
                self.inherit,
                self.parent,
            ),
        ]

    @staticmethod
    def _evaluate(value: Chance, namespace: SimpleNamespace) -> object:
        """Return a number, or the value of a function of the namespace."""
        return value(namespace) if callable(value) else value

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the requests of the action where the condition holds.

        They fire at the rate or with the probability.

        Args:
            namespace:
                The namespace of an edge, with `namespace.source` and `namespace.target`.

        Returns:
            laws:
                A jump law of the request at a rate, or else an update law of the request with its probability.
        """
        chance: object = 1
        if self.rate is not None:
            chance = self._evaluate(self.rate, namespace)
        elif self.probability is not None:
            chance = self._evaluate(self.probability, namespace)
        if self.condition is not None:
            chance = where(self.condition(namespace), chance, 0)
        if self.rate is not None:
            return [JumpLaw(chance, {request: 1}) for request in self._get_requests()]
        return [UpdateLaw(request, chance) for request in self._get_requests()]
