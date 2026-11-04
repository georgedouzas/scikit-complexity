"""Define topology changes, which add and remove the nodes and edges of a network and rewire its edges."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Annotated

from ...errors import ValidationError
from ...specs import BaseSpec, RoleSpec, TopologyRequestSpec
from ..laws import BaseLaw, JumpLaw, UpdateLaw
from ..laws.utils import where
from ._neighbor import NeighborInteraction

ACTIONS = ('add-node', 'remove-node', 'replace-node', 'add-edge', 'remove-edge', 'rewire')

type Chance = float | Callable[[SimpleNamespace], object]


class TopologyInteraction(NeighborInteraction):
    """Change of the topology of a network: nodes and edges that are added or removed, and edges that are rewired.

    The action runs for every node of the group a role binds, `'add-node'`, `'remove-node'`, `'replace-node'`, which
    removes the node and adds a new one, and `'add-edge'`, or for every edge of its network, `'remove-edge'` and
    `'rewire'`. Without a target, `'add-node'` runs once per step, adding one node from outside the network to the one
    network whose groups are variable, as in growing networks. It fires at a `rate`, with a `probability` per step, or
    with probability one, and only where the `condition` holds. The rate, the probability, and the condition are numbers
    or functions of the namespace of an edge, whose ends are `namespace.source` and `namespace.target`, and an action of
    the nodes reads the node as the source. Without a target, the namespace of `'add-node'` holds the values drawn for
    the node being added as `namespace.source`, which the condition and a probability read and a rate does not. A new
    node joins `edges` other nodes, a node adding edges joins `edges` other nodes, and a rewiring keeps the end `keep`
    of its edge. New ends are drawn with probability proportional to `toward(node)`, a function of the namespace of a
    candidate node that also holds the node the new edge starts from as `node.source`, or uniformly. That node is the
    new node of `'add-node'`, the adding node of `'add-edge'`, and the kept end of `'rewire'`. Where fewer candidates
    have a positive weight than the edges to draw, the new ends are drawn uniformly. A new node or edge takes the
    `values`, each a value or a function of the random number generator that draws it. The groups on the nodes and the
    edges of the network are variable. With `hyperedge`, a new node instead forms a hyperedge of that size with
    `hyperedge - 1` existing nodes and joins each of them by an edge: the first two are the ends of an edge drawn
    uniformly, so a size of three attaches a triangle along an edge, and further members are drawn by `toward`. The
    new hyperedge is a new member of the variable group on the hyperedges of the network.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the nodes of the network, or `None` for an addition of one node per step.

        action:
            The change, `'add-node'`, `'remove-node'`, `'replace-node'`, `'add-edge'`, `'remove-edge'`, or
            `'rewire'`.

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
            The size of the hyperedge every new node forms with existing nodes, or `None` for none.

    Raises:
        ValidationError:
            The action or the kept end is unknown, both a rate and a probability are given, the number of edges is
            not a whole number, an action other than `'add-node'` has no target, or a hyperedge is given to an action
            other than `'add-node'` or has fewer than two members, raised by `check`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.elements import Element, GroupElement
        >>> from skcomplex.core.interactions import TopologyInteraction
        >>> from skcomplex.core.spaces import NetworkSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=20)
        >>> links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
        >>> expiry = TopologyInteraction('expiry', target='nodes', action='remove-edge')
        >>> system = System(elements=[nodes, links], interactions=[expiry], space=NetworkSpace(nx.cycle_graph(5)))
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
        >>> len(trajectory.network(0).sources), len(trajectory.network(1).sources)
        (5, 0)

        A network that grows by triangles from one triangle:

        >>> import xgi
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> triangles = GroupElement('triangles', element=Element, on='hyperedges', variable=True, capacity=8)
        >>> growth = TopologyInteraction('growth', action='add-node', hyperedge=3)
        >>> space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
        >>> system = System(
        ...     elements=[nodes, links, triangles], interactions=[growth], space=space, time=DiscreteTime()
        ... )
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=5).trajectory_
        >>> trajectory.network(-1).hyperedges.shape
        (6, 3)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str | None, RoleSpec(optional=True)] = None,
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

    def _check(self) -> None:
        """Raise an error for an unknown action or kept end, a rate next to a probability, or a bad number of edges.

        Raises:
            ValidationError:
                The action or the kept end is unknown, both a rate and a probability are given, a condition or a
                weight cannot be called, the number of edges is not a whole number of at least 0, an action other
                than `'add-node'` has no target, or a hyperedge is given to an action other than `'add-node'`.
        """
        super()._check()
        self._check_variant('action', self.action, ACTIONS, (BaseSpec,))
        if self.rate is not None and self.probability is not None:
            error_msg = f'{self.label!r} fires at a rate and with a probability. Give one of them.'
            raise ValidationError(error_msg)
        if self.target is None and self.action != 'add-node':
            error_msg = f'{self.label!r} runs {self.action!r} without a target. Give the group on the nodes.'
            raise ValidationError(error_msg)
        if self.hyperedge is not None and self.action != 'add-node':
            error_msg = f'{self.label!r} runs {self.action!r} with a hyperedge. Only new nodes form hyperedges.'
            raise ValidationError(error_msg)
        for name in ('condition', 'toward'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace, or None.'
                raise ValidationError(error_msg)
        for request in self._get_requests():
            request.check()

    def _get_requests(self) -> list[TopologyRequestSpec]:
        """Return the requests of the action, two for a replacement of a node."""
        if self.action not in ACTIONS:
            raise self._get_variant_error(self.action, 'action', '_get_requests')
        values = tuple((self.values or {}).items())
        kinds = ['add-node', 'remove-node'] if self.action == 'replace-node' else [self.action]
        return [
            TopologyRequestSpec(
                kind,
                self.edges,
                self.toward,
                self.keep,
                self.layer,
                values,
                self.hyperedge if kind == 'add-node' else None,
            )
            for kind in kinds
        ]

    @staticmethod
    def _evaluate(value: Chance, namespace: SimpleNamespace) -> object:
        """Return a number, or the value of a function of the namespace."""
        return value(namespace) if callable(value) else value

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the requests of the action, fired at the rate or with the probability where the condition holds.

        Args:
            namespace:
                The namespace of an edge, with `namespace.source` and `namespace.target`.

        Returns:
            laws:
                A jump law per request at a rate, or else an update law per request with its probability.
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
