"""Apply the topology requests of laws to the capacity rows of the groups of a network between the steps of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..base import CapacityError
from ..laws import TopologyRequest
from ._backends import reads_reductions
from ._kernels import Kernel
from ._layout import Entry
from ._neighbourhoods import ACTIVE_THRESHOLD
from ._problem import NetworkSnapshot, Vector
from ._reductions import ReductionPass

type Indices = npt.NDArray[np.intp]
type Pair = tuple[int, int]


@dataclass(frozen=True)
class CapacityRows:
    """Rows of a variable group on the nodes or edges of a network, with the values a new member starts from.

    Args:
        path:
            The path of the group.

        variables:
            The entry of every state and parameter of the members by name, with whether it is a state.

        defaults:
            The flat values of one new member of every variable by name.

        active:
            The parameter entry of the activity of every row.

        source:
            The parameter entry of the source node of every edge row, or `None` for a group on the nodes.

        target:
            The parameter entry of the target node of every edge row, or `None` for a group on the nodes.
    """

    path: str
    variables: Mapping[str, tuple[Entry, bool]]
    defaults: Mapping[str, Vector]
    active: Entry
    source: Entry | None = None
    target: Entry | None = None

    def present(self, p: Vector) -> npt.NDArray[np.bool_]:
        """Return whether every row holds a member.

        Args:
            p:
                The parameter vector.

        Returns:
            present:
                One flag per row.
        """
        return self.active.view(p)[:, 0] > ACTIVE_THRESHOLD

    def take_row(self, p: Vector) -> int:
        """Return the first free row.

        Args:
            p:
                The parameter vector.

        Returns:
            row:
                The row.

        Raises:
            CapacityError:
                Every row holds a member.
        """
        free = np.flatnonzero(~self.present(p))
        if not free.size:
            error_msg = (
                f'The group {self.path} holds {self.active.rows} members, its capacity, and a topology request adds '
                'another. Give the group a larger capacity, `Group(..., variable=True, capacity=n)`.'
            )
            raise CapacityError(error_msg)
        return int(free[0])

    def write(self, row: int, x: Vector, p: Vector, values: Mapping[str, object]) -> None:
        """Start a member in a row with given values or the defaults, and mark the row present.

        Args:
            row:
                The row.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new member by variable name, which override the defaults.
        """
        for name, (entry, is_state) in self.variables.items():
            vector = x if is_state else p
            value = values.get(name, self.defaults[name])
            start = entry.start + row * entry.columns
            vector[start : start + entry.columns] = np.broadcast_to(np.asarray(value, dtype=float), (entry.columns,))
        p[self.active.start + row] = 1.0

    def clear(self, row: int, p: Vector) -> None:
        """Mark a row free.

        Args:
            row:
                The row.

            p:
                The parameter vector, changed in place.
        """
        p[self.active.start + row] = 0.0

    def ends(self, p: Vector) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge row.

        Args:
            p:
                The parameter vector.

        Returns:
            ends:
                The source and target node rows.
        """
        assert self.source is not None  # nosec B101: called for edge groups only
        assert self.target is not None  # nosec B101: called for edge groups only
        return self.source.view(p)[:, 0].astype(np.intp), self.target.view(p)[:, 0].astype(np.intp)


@dataclass(frozen=True)
class NetworkRows:
    """Variable groups of one network: the groups on its nodes and the edge group of every changing layer.

    Args:
        nodes:
            The groups on the nodes, which share their rows.

        edges:
            The variable edge group of every layer by label, `None` for the graph of the network.

        directed:
            Whether the edges of every layer are directed.
    """

    nodes: tuple[CapacityRows, ...]
    edges: Mapping[str | None, CapacityRows]
    directed: Mapping[str | None, bool]

    def present(self, p: Vector) -> npt.NDArray[np.bool_]:
        """Return whether every node row holds a node.

        Args:
            p:
                The parameter vector.

        Returns:
            present:
                One flag per node row.
        """
        return self.nodes[0].present(p)

    def order_ends(self, layer: str | None, first: int, second: int) -> Pair:
        """Return the ends of an edge, in node order on an undirected layer.

        Args:
            layer:
                The layer.

            first:
                The node row of one end.

            second:
                The node row of the other end.

        Returns:
            ends:
                The source and target node rows.
        """
        return (first, second) if self.directed[layer] else (min(first, second), max(first, second))

    def neighbours(self, layer: str | None, node: int, p: Vector) -> Indices:
        """Return the node rows a node joins in a layer, its successors on a directed layer.

        Args:
            layer:
                The layer.

            node:
                The node row.

            p:
                The parameter vector.

        Returns:
            neighbours:
                The node rows.
        """
        edges = self.edges[layer]
        sources, targets = edges.ends(p)
        present = edges.present(p)
        found = targets[present & (sources == node)]
        if self.directed[layer]:
            return found
        return np.concatenate([found, sources[present & (targets == node)]])

    def add_node(self, x: Vector, p: Vector, values: Mapping[str, object]) -> int:
        """Add a node in the first free node row of every group on the nodes.

        Args:
            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new node by variable name.

        Returns:
            row:
                The node row.
        """
        row = self.nodes[0].take_row(p)
        for group in self.nodes:
            group.write(row, x, p, values)
        return row

    def add_edge(self, layer: str | None, ends: Pair, x: Vector, p: Vector, values: Mapping[str, object]) -> None:
        """Add an edge between two node rows to a layer.

        Args:
            layer:
                The layer.

            ends:
                The source and target node rows.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new edge by variable name.
        """
        edges = self.edges[layer]
        row = edges.take_row(p)
        first, second = self.order_ends(layer, *ends)
        edges.write(row, x, p, values)
        p[edges.source.start + row] = first  # type: ignore[union-attr]
        p[edges.target.start + row] = second  # type: ignore[union-attr]

    def remove_node(self, row: int, p: Vector) -> None:
        """Remove a node and every edge that joins it.

        Args:
            row:
                The node row.

            p:
                The parameter vector, changed in place.
        """
        for group in self.nodes:
            group.clear(row, p)
        for edges in self.edges.values():
            sources, targets = edges.ends(p)
            for edge in np.flatnonzero(edges.present(p) & ((sources == row) | (targets == row))):
                edges.clear(int(edge), p)

    def snapshot(self, p: Vector, layer: str | None) -> NetworkSnapshot:
        """Return the nodes and the edges of a layer present at the parameters of a frame.

        Args:
            p:
                The parameter vector of the frame.

            layer:
                The layer.

        Returns:
            snapshot:
                The nodes and edges present.
        """
        edges = self.edges[layer]
        present = edges.present(p)
        sources, targets = edges.ends(p)
        return NetworkSnapshot(
            np.flatnonzero(self.present(p)).astype(np.intp),
            sources[present],
            targets[present],
            np.flatnonzero(present).astype(np.intp),
        )


@dataclass(frozen=True)
class CompiledRequest:
    """Topology request of a law with the rows it runs for and the kernels that decide and direct it.

    Args:
        request:
            The request.

        owner:
            The path of the component whose law requests it.

        network:
            The variable groups of the network it changes.

        rows:
            `'node'` for a law of the nodes, `'edge'` for a law of the edges of a layer, or `'single'` for a law that
            runs once.

        condition:
            The kernel of the probability that an update law fires the request for every row, or `None` for a jump.

        toward:
            The kernel of the weight of every node row as a new end, or `None` for equal weights.
    """

    request: TopologyRequest
    owner: str
    network: NetworkRows
    rows: str
    condition: Kernel | None = None
    toward: Kernel | None = None


@dataclass
class Topology:
    """Topology requests of a problem, applied between steps to the rows of the variable groups of its networks.

    Args:
        networks:
            The variable groups of every changing network.

        requests:
            The requests of update laws, in the order they fire.

        channels:
            The requests of jump laws by the index of their jump channels.

        reductions:
            The reductions the kernels of the requests read.
    """

    networks: tuple[NetworkRows, ...]
    requests: tuple[CompiledRequest, ...] = ()
    channels: dict[int, CompiledRequest] = field(default_factory=dict)
    reductions: ReductionPass = field(default_factory=ReductionPass)

    @property
    def layers(self) -> tuple[tuple[str, str | None], ...]:
        """Path of the first group on the nodes and the layer of every network layer with a variable edge group."""
        return tuple((network.nodes[0].path, layer) for network in self.networks for layer in network.edges)

    def decide(self, t: float, x: Vector, p: Vector, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return the requests of update laws that fire at a state, each as its index and the row it fires for.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

            rng:
                The random number generator.

        Returns:
            fired:
                The index of every fired request with its row, in the order of the requests and the rows.
        """
        if not self.requests:
            return []
        conditions = [item.condition for item in self.requests if item.condition is not None]
        q = self.reductions.extend(t, x, p) if reads_reductions(conditions) else p
        fired = []
        for index, item in enumerate(self.requests):
            assert item.condition is not None  # nosec B101: update requests carry their condition
            _, _, values = item.condition.evaluate(t, x, q)
            chances = np.clip(np.asarray(values[0], dtype=float), 0.0, 1.0)
            rows = np.flatnonzero(rng.random(chances.size) < chances)
            fired += [(index, int(row)) for row in rows]
        return fired

    def apply(
        self,
        fired: list[tuple[int, int]],
        t: float,
        x: Vector,
        p: Vector,
        rng: np.random.Generator,
    ) -> Vector:
        """Apply fired requests of update laws to a state and to the parameters in place, and return the state.

        Args:
            fired:
                The fired requests, as `decide` returns them.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector, changed in place.

            rng:
                The random number generator.

        Returns:
            state:
                The state after the requests.
        """
        state = x.copy()
        for index, row in fired:
            self._execute(self.requests[index], row, t, state, p, rng=rng)
        return state

    def fire(self, channel: int, row: int, t: float, x: Vector, p: Vector, *, rng: np.random.Generator) -> Vector:
        """Apply the request of a jump channel for a row, and return the state.

        Args:
            channel:
                The index of the jump channel.

            row:
                The row the jump fires for.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector, changed in place.

            rng:
                The random number generator.

        Returns:
            state:
                The state after the request.
        """
        state = x.copy()
        self._execute(self.channels[channel], row, t, state, p, rng=rng)
        return state

    def read_network(self, p: Vector, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshot:
        """Return the nodes and edges of a layer of a network at the parameters of a frame.

        Args:
            p:
                The parameter vector of the frame.

            nodes:
                The path of a group on the nodes of the network, or `None` for the first network.

            layer:
                The layer.

        Returns:
            snapshot:
                The nodes and edges present.
        """
        network = next(
            (item for item in self.networks if nodes is None or nodes in {group.path for group in item.nodes}),
            self.networks[0],
        )
        return network.snapshot(p, layer)

    def presence(self, name: str, p: Vector) -> npt.NDArray[np.bool_] | None:
        """Return whether every row of the group of a variable holds a member, or `None` for a fixed group.

        Args:
            name:
                The dotted name of a state or parameter.

            p:
                The parameter vector of a frame.

        Returns:
            present:
                One flag per row of the group, or `None`.
        """
        for network in self.networks:
            for group in (*network.nodes, *network.edges.values()):
                if any(entry.name == name for entry, _ in group.variables.values()):
                    return group.present(p)
        return None

    def _weights(self, item: CompiledRequest, t: float, x: Vector, p: Vector) -> Vector:
        """Return the weight of every node row as a new end of a request."""
        size = item.network.nodes[0].active.rows
        if item.toward is None:
            return np.ones(size)
        _, _, values = item.toward.evaluate(t, x, self.reductions.extend(t, x, p))
        weights = np.broadcast_to(np.asarray(values[0], dtype=float), (size,))
        return np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    @staticmethod
    def _draw(weights: Vector, allowed: npt.NDArray[np.bool_], count: int, rng: np.random.Generator) -> list[int]:
        """Return up to `count` distinct allowed rows drawn with probabilities proportional to their weights."""
        candidates = np.flatnonzero(allowed)
        if not candidates.size or count <= 0:
            return []
        chances = weights[candidates]
        chances = chances / chances.sum() if chances.sum() > 0 else np.full(candidates.size, 1 / candidates.size)
        positive = int(np.count_nonzero(chances))
        if positive < min(count, candidates.size):
            chances = np.full(candidates.size, 1 / candidates.size)
        chosen = rng.choice(candidates, size=min(count, candidates.size), replace=False, p=chances)
        return [int(item) for item in np.atleast_1d(chosen)]

    @staticmethod
    def _allowed(item: CompiledRequest, end: int, p: Vector) -> npt.NDArray[np.bool_]:
        """Return the node rows present that a node may join, other than itself and its neighbours in the layer."""
        allowed = item.network.present(p).copy()
        allowed[end] = False
        allowed[item.network.neighbours(item.request.layer, end, p)] = False
        return allowed

    def _join(
        self,
        item: CompiledRequest,
        end: int,
        t: float,
        x: Vector,
        p: Vector,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Join a node to new ends drawn by the weights of a request, with the values of an edge request."""
        request, network = item.request, item.network
        allowed = self._allowed(item, end, p)
        values = dict(request.values) if request.kind == 'add-edge' else {}
        for other in self._draw(self._weights(item, t, x, p), allowed, request.edges, rng):
            network.add_edge(request.layer, (end, other), x, p, values)

    def _rewire(
        self,
        item: CompiledRequest,
        row: int,
        t: float,
        x: Vector,
        p: Vector,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Move the free end of an edge row to a node drawn by the weights of a request."""
        request, network = item.request, item.network
        edges = network.edges[request.layer]
        sources, targets = edges.ends(p)
        kept = int(sources[row] if request.keep == 'source' else targets[row])
        allowed = self._allowed(item, kept, p)
        chosen = self._draw(self._weights(item, t, x, p), allowed, 1, rng)
        if not chosen:
            return
        ends = (kept, chosen[0]) if request.keep == 'source' else (chosen[0], kept)
        first, second = network.order_ends(request.layer, *ends)
        p[edges.source.start + row] = first  # type: ignore[union-attr]
        p[edges.target.start + row] = second  # type: ignore[union-attr]

    def _execute(
        self,
        item: CompiledRequest,
        row: int,
        t: float,
        x: Vector,
        p: Vector,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Apply one request for one row, skipping a row whose node or edge is no longer present."""
        request, network = item.request, item.network
        if item.rows == 'edge' and not network.edges[request.layer].present(p)[row]:
            return
        if item.rows == 'node' and not network.present(p)[row]:
            return
        kind = request.kind
        if kind == 'add-node':
            node = network.add_node(x, p, dict(request.values))
            self._join(item, node, t, x, p, rng=rng)
        elif kind == 'add-edge':
            self._join(item, row, t, x, p, rng=rng)
        elif kind == 'remove-node':
            network.remove_node(row, p)
        elif kind == 'remove-edge':
            network.edges[request.layer].clear(row, p)
        else:
            self._rewire(item, row, t, x, p, rng=rng)
