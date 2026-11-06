"""Apply the topology requests of laws to the capacity rows of the groups of a network between the steps of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from ...components.laws import JumpLaw, UpdateLaw
from ...errors import ValidationError
from ...specs import (
    EDGE_KINDS,
    CapacityRowsSpec,
    NetworkRowsSpec,
    NetworkSnapshotSpec,
    RunStateSpec,
    TopologyRequestSpec,
)
from ...types import Array, Pair
from ._base import BaseKernel
from ._compiled_request_spec import CompiledRequestSpec
from ._reduction_schedule_spec import ReductionScheduleSpec

_NO_REDUCTIONS = ReductionScheduleSpec()
MIN_EDGE_HYPEREDGE = 2

type Fired = tuple[int, int, Mapping[str, object] | None]


class TopologyKernel(BaseKernel):
    """Topology requests of a problem, applied between steps to the rows of the variable groups of its networks.

    `run(state)` fires the requests of update laws at the state and keeps the state after them in `values_`. The solver
    of jump laws applies the request of every jump channel that fires. The weights of a request read the node its new
    ends join as the source, and a condition that reads the source of an addition draws the new node before it decides,
    so a node that does not fire is never added and a node that fires keeps those values. A new node of a request with
    a hyperedge forms a new member of the variable group on the hyperedges, with the ends of an edge it draws and
    further nodes drawn by the weights of the request, and joins each of them by an edge, or, without edges to draw or
    with independent ends, with nodes drawn by the weights alone. A node of `'add-hyperedge'` forms a new hyperedge
    with nodes drawn by the weights, and `'remove-hyperedge'` frees the row of a hyperedge. A removed node takes its
    hyperedges with it. Independent ends join every candidate on its own with the weight as its probability.

    Args:
        networks:
            The variable groups of every changing network.

        requests:
            The requests of update laws, in the order they fire.

        channels:
            The requests of jump laws by the index of their jump channels.

        reductions:
            The reductions the kernels of the requests read.

    Attributes:
        values_:
            The state vector after the requests of the last run.
    """

    values_: Array

    def __init__(
        self,
        networks: tuple[NetworkRowsSpec, ...],
        requests: tuple[CompiledRequestSpec, ...] = (),
        channels: Mapping[int, CompiledRequestSpec] | None = None,
        reductions: ReductionScheduleSpec = _NO_REDUCTIONS,
    ) -> None:
        self.networks = networks
        self.requests = requests
        self.channels = channels
        self.reductions = reductions

    @property
    def layers(self) -> tuple[tuple[str, str | None], ...]:
        """Path of the first group on the nodes and the layer of every network layer with a variable edge group."""
        return tuple((network.nodes[0].path, layer) for network in self.networks for layer in network.edges)

    def _decide(self, t: float, x: Array, p: Array, rng: np.random.Generator) -> list[Fired]:
        """Return the requests of update laws that fire at a state, as an index, a row, and values drawn to decide.

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
                The index of every fired request with its row, in the order of the requests and the rows, and the
                values of the new node when the condition read them, or else `None`.
        """
        if not self.requests:
            return []
        conditions = [item.condition for item in self.requests if item.condition is not None]
        if len(conditions) != len(self.requests):
            error_msg = 'Every update request of a topology kernel carries its condition.'
            raise ValidationError(error_msg)
        q = self.reductions.extend(t, x, p) if any(item.reads_reductions for item in conditions) else p
        fired: list[Fired] = []
        for index, condition in enumerate(conditions):
            item = self.requests[index]
            drawn = self._initial_values(item.request, rng) if condition.reads_source else None
            source = None if drawn is None else self._new_values(item.network.nodes[0], drawn)
            _, _, values = condition.run(RunStateSpec(t, x, q, source=source)).values_
            chances = np.clip(np.asarray(values[0], dtype=float), 0.0, 1.0)
            rows = np.flatnonzero(rng.random(chances.size) < chances)
            fired += [(index, int(row), drawn) for row in rows]
        return fired

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Fire the requests of update laws at the state, apply them, and set the state after them.

        The requests decide at the state which rows fire, and the fired requests change a copy of the state vector and
        the parameter vector of the state in place.
        """
        state = self._state_of(inputs)
        t, x, p, rng = state.t, state.x, state.p, self._generator_of(state)
        self.values_ = self._apply(self._decide(t, x, p, rng), t, x, p, rng)

    def _apply(self, fired: list[Fired], t: float, x: Array, p: Array, rng: np.random.Generator) -> Array:
        """Apply fired requests of update laws to a copy of a state and to the parameters in place."""
        state = x.copy()
        for index, row, drawn in fired:
            self._execute(self.requests[index], row, t, state, p, rng=rng, drawn=drawn)
        return state

    def _fire(self, channel: int, row: int, t: float, x: Array, p: Array, *, rng: np.random.Generator) -> Array:
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
        self._execute((self.channels or {})[channel], row, t, state, p, rng=rng)
        return state

    def read_network(self, p: Array, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshotSpec:
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

    @property
    def variable_names(self) -> frozenset[str]:
        """Dotted names of the states and parameters of the variable groups, whose rows hold no fixed member."""
        return frozenset(
            entry.name
            for network in self.networks
            for group in (
                *network.nodes,
                *network.edges.values(),
                *(() if network.hyperedges is None else (network.hyperedges,)),
            )
            for entry, _ in group.variables.values()
        )

    def presence(self, name: str, p: Array) -> npt.NDArray[np.bool_] | None:
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
            hyperedges = () if network.hyperedges is None else (network.hyperedges,)
            for group in (*network.nodes, *network.edges.values(), *hyperedges):
                if any(entry.name == name for entry, _ in group.variables.values()):
                    return group.present(p)
        return None

    def _weights(self, item: CompiledRequestSpec, end: int, t: float, x: Array, p: Array) -> Array:
        """Return the weight of every node row as a new end of a request whose edges start from a node row."""
        nodes = item.network.nodes[0]
        size = nodes.active.rows
        if item.toward is None:
            return np.ones(size)
        source = self._row_values(nodes, end, x, p)
        _, _, values = item.toward.run(RunStateSpec(t, x, self.reductions.extend(t, x, p), source=source)).values_
        weights = np.broadcast_to(np.asarray(values[0], dtype=float), (size,))
        return np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    @staticmethod
    def _draw(weights: Array, allowed: npt.NDArray[np.bool_], count: int, rng: np.random.Generator) -> list[int]:
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
    def _allowed(item: CompiledRequestSpec, end: int, p: Array, *, incoming: bool = False) -> npt.NDArray[np.bool_]:
        """Return the node rows present that a node may join, other than itself and its neighbours in the layer.

        On a directed layer the neighbours are the successors of the node, or its predecessors for edges into it.
        """
        network, layer = item.network, item.request.layer
        allowed = network.present(p).copy()
        allowed[end] = False
        if incoming and network.directed[layer]:
            edges = network.edges[layer]
            sources, targets = edges.ends(p)
            allowed[sources[edges.present(p) & (targets == end)]] = False
        else:
            allowed[network.neighbours(layer, end, p)] = False
        return allowed

    @staticmethod
    def _independent(weights: Array, allowed: npt.NDArray[np.bool_], rng: np.random.Generator) -> list[int]:
        """Return the allowed rows that each join on their own with their weight, clipped to `[0, 1]`, as probability.

        Args:
            weights:
                The weight of every node row.

            allowed:
                Whether every node row may join.

            rng:
                The random number generator.

        Returns:
            rows:
                The rows that join, in increasing order.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.problems._topology_kernel import TopologyKernel
            >>> weights, allowed = np.array([1.0, 0.0, 2.0, 1.0]), np.array([True, True, True, False])
            >>> TopologyKernel._independent(weights, allowed, np.random.default_rng(0))
            [0, 2]
        """
        candidates = np.flatnonzero(allowed)
        chances = np.clip(weights[candidates], 0.0, 1.0)
        return [int(row) for row in candidates[rng.random(candidates.size) < chances]]

    def _join(
        self,
        item: CompiledRequestSpec,
        end: int,
        t: float,
        x: Array,
        p: Array,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Join a node to new ends drawn by the weights of a request, with the values of an edge request.

        The new edges leave the node, enter it, or go both ways on a directed layer, by the direction of the request.
        Independent ends join every candidate on its own, apart for each way.
        """
        request, network = item.request, item.network
        weights = self._weights(item, end, t, x, p)
        directions = ['out', 'in'] if request.direction == 'both' else [request.direction]
        if request.independent:
            pairs = [
                (end, other) if direction == 'out' else (other, end)
                for direction in directions
                for other in self._independent(weights, self._allowed(item, end, p, incoming=direction == 'in'), rng)
            ]
        else:
            allowed = self._allowed(item, end, p, incoming=request.direction == 'in')
            drawn = self._draw(weights, allowed, request.edges, rng)
            pairs = [
                (end, other) if direction == 'out' else (other, end) for other in drawn for direction in directions
            ]
        for ends in pairs:
            values = self._initial_values(request, rng) if request.kind == 'add-edge' else {}
            self._add_edge(network, request.layer, ends, x, p, values=values)

    def _attach(
        self,
        item: CompiledRequestSpec,
        node: int,
        t: float,
        x: Array,
        p: Array,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Form a hyperedge of a node with other nodes, drawn by the weights of a request, after the ends of an edge.

        A new node with edges to draw and no independent ends joins every other member of the hyperedge by an edge,
        and the hyperedge lists the ends of a drawn edge first and the new node last, so an edge joins its first two
        members. Otherwise every other member is drawn by the weights, the node is the last member, and a new node
        draws its edges apart. A network with too few nodes for the hyperedge forms none.

        Args:
            item:
                The compiled request, whose request has a hyperedge.

            node:
                The node row of the new node or of the node that forms the hyperedge.

            t:
                The time.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            rng:
                The random number generator.
        """
        request, network = item.request, item.network
        size = int(request.hyperedge or 0)
        anchored = request.kind == 'add-node' and request.edges > 0 and not request.independent
        members: list[int] = []
        if anchored and size > MIN_EDGE_HYPEREDGE:
            edges = network.edges[request.layer]
            present = np.flatnonzero(edges.present(p))
            if present.size:
                sources, targets = edges.ends(p)
                row = int(rng.choice(present))
                members = [int(sources[row]), int(targets[row])]
        allowed = network.present(p).copy()
        allowed[[node, *members]] = False
        members += self._draw(self._weights(item, node, t, x, p), allowed, size - 1 - len(members), rng)
        if len(members) == size - 1:
            if anchored:
                for other in members:
                    self._add_edge(network, request.layer, (other, node), x, p, values={})
            self._add_hyperedge(item, [*members, node], x, p)
        if request.kind == 'add-node' and not anchored:
            self._join(item, node, t, x, p, rng=rng)

    def _add_hyperedge(self, item: CompiledRequestSpec, members: list[int], x: Array, p: Array) -> None:
        """Write a new hyperedge of member node rows into the first free row of the variable group on hyperedges.

        Args:
            item:
                The compiled request.

            members:
                The node rows of the members, in hyperedge order.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

        Raises:
            ValidationError:
                The network has no variable group on its hyperedges, or the hyperedge holds more members than its
                rows have columns.
        """
        hyperedges = item.network.hyperedges
        if hyperedges is None or hyperedges.members is None:
            error_msg = f'{item.owner} forms hyperedges on a network without a variable group on its hyperedges.'
            raise ValidationError(error_msg)
        width = hyperedges.members.columns
        if len(members) > width:
            error_msg = f'{item.owner} forms a hyperedge of {len(members)} members, but the rows hold {width}.'
            raise ValidationError(error_msg)
        row = hyperedges.take_row(p)
        self._write(hyperedges, row, x, p, {})
        start = hyperedges.members.start + row * width
        p[start : start + width] = [*members, *([-1] * (width - len(members)))]

    def _rewire(
        self,
        item: CompiledRequestSpec,
        row: int,
        t: float,
        x: Array,
        p: Array,
        *,
        rng: np.random.Generator,
    ) -> None:
        """Move the free end of an edge row to a node drawn by the weights of a request."""
        request, network = item.request, item.network
        edges = network.edges[request.layer]
        sources, targets = edges.ends(p)
        kept = int(sources[row] if request.keep == 'source' else targets[row])
        allowed = self._allowed(item, kept, p)
        chosen = self._draw(self._weights(item, kept, t, x, p), allowed, 1, rng)
        if not chosen:
            return
        ends = (kept, chosen[0]) if request.keep == 'source' else (chosen[0], kept)
        first, second = network.order_ends(request.layer, *ends)
        self._set_ends(edges, row, (first, second), p)

    def _execute(
        self,
        item: CompiledRequestSpec,
        row: int,
        t: float,
        x: Array,
        p: Array,
        *,
        rng: np.random.Generator,
        drawn: Mapping[str, object] | None = None,
    ) -> None:
        """Apply one request for one row, skipping a row whose node, edge, or hyperedge is no longer present."""
        request, network = item.request, item.network
        if not self._holds(item, row, p):
            return
        hyperedges = network.hyperedges
        kind = request.kind
        if kind == 'add-node':
            values = self._initial_values(request, rng) if drawn is None else drawn
            node = self._add_node(network, x, p, values)
            if request.hyperedge is None:
                self._join(item, node, t, x, p, rng=rng)
            else:
                self._attach(item, node, t, x, p, rng=rng)
        elif kind == 'add-edge':
            self._join(item, row, t, x, p, rng=rng)
        elif kind == 'add-hyperedge':
            self._attach(item, row, t, x, p, rng=rng)
        elif kind == 'remove-hyperedge':
            if hyperedges is not None:
                self._clear(hyperedges, row, p)
        elif kind == 'remove-node':
            self._remove_node(network, row, p)
        elif kind == 'remove-edge':
            self._clear(network.edges[request.layer], row, p)
        else:
            self._rewire(item, row, t, x, p, rng=rng)

    @staticmethod
    def _holds(item: CompiledRequestSpec, row: int, p: Array) -> bool:
        """Return whether the node, edge, or hyperedge of the row a request runs for is present, `True` for one run.

        Args:
            item:
                The compiled request.

            row:
                The row it runs for.

            p:
                The parameter vector.

        Returns:
            present:
                Whether the row holds a member.
        """
        network = item.network
        if item.rows == 'edge':
            return bool(network.edges[item.request.layer].present(p)[row])
        if item.rows == 'node':
            return bool(network.present(p)[row])
        if item.rows == 'hyperedge':
            return network.hyperedges is not None and bool(network.hyperedges.present(p)[row])
        return True

    @staticmethod
    def _initial_values(request: TopologyRequestSpec, rng: np.random.Generator) -> dict[str, object]:
        """Return the values of a new node or edge of a request, drawing those given as functions of the generator."""
        return {name: value(rng) if callable(value) else value for name, value in request.values}

    @staticmethod
    def _row_values(group: CapacityRowsSpec, row: int, x: Array, p: Array) -> dict[str, Array]:
        """Return the values of the member in a row of a group by dotted name."""
        values = {}
        for entry, is_state in group.variables.values():
            start = entry.start + row * entry.columns
            values[entry.name] = (x if is_state else p)[start : start + entry.columns]
        return values

    @staticmethod
    def _new_values(group: CapacityRowsSpec, values: Mapping[str, object]) -> dict[str, Array]:
        """Return the values a new member of a group would take by dotted name, given values or the defaults."""
        return {
            entry.name: np.broadcast_to(np.asarray(values.get(name, group.defaults[name]), dtype=float), entry.columns)
            for name, (entry, _) in group.variables.items()
        }

    @staticmethod
    def _write(group: CapacityRowsSpec, row: int, x: Array, p: Array, values: Mapping[str, object]) -> None:
        """Start a member in a row with given values or the defaults, and mark the row present.

        Args:
            group:
                The variable group.

            row:
                The row.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new member by variable name, which override the defaults.
        """
        for name, (entry, is_state) in group.variables.items():
            vector = x if is_state else p
            value = values.get(name, group.defaults[name])
            start = entry.start + row * entry.columns
            vector[start : start + entry.columns] = np.broadcast_to(np.asarray(value, dtype=float), (entry.columns,))
        p[group.active.start + row] = 1.0

    @staticmethod
    def _clear(group: CapacityRowsSpec, row: int, p: Array) -> None:
        """Mark a row free.

        Args:
            group:
                The variable group.

            row:
                The row.

            p:
                The parameter vector, changed in place.
        """
        p[group.active.start + row] = 0.0

    def _add_node(self, network: NetworkRowsSpec, x: Array, p: Array, values: Mapping[str, object]) -> int:
        """Add a node in the first free node row of every group on the nodes.

        Args:
            network:
                The network.

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
        row = network.nodes[0].take_row(p)
        for group in network.nodes:
            self._write(group, row, x, p, values)
        return row

    def _add_edge(
        self,
        network: NetworkRowsSpec,
        layer: str | None,
        ends: Pair,
        x: Array,
        p: Array,
        *,
        values: Mapping[str, object],
    ) -> None:
        """Add an edge between two node rows to a layer.

        Args:
            network:
                The network.

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
        edges = network.edges[layer]
        row = edges.take_row(p)
        first, second = network.order_ends(layer, *ends)
        self._write(edges, row, x, p, values)
        self._set_ends(edges, row, (first, second), p)

    @staticmethod
    def _set_ends(edges: CapacityRowsSpec, row: int, ends: Pair, p: Array) -> None:
        """Write the source and target node of an edge row into the parameter vector.

        Args:
            edges:
                The rows of an edge group.

            row:
                The edge row.

            ends:
                The source and target node rows.

            p:
                The parameter vector, changed in place.

        Raises:
            ValidationError:
                The group is a group on the nodes, whose rows have no ends.
        """
        if edges.source is None or edges.target is None:
            error_msg = 'Only the rows of an edge group have a source and a target node.'
            raise ValidationError(error_msg)
        p[edges.source.start + row], p[edges.target.start + row] = ends

    def _remove_node(self, network: NetworkRowsSpec, row: int, p: Array) -> None:
        """Remove a node with every edge that joins it and every hyperedge that holds it.

        Args:
            network:
                The network.

            row:
                The node row.

            p:
                The parameter vector, changed in place.
        """
        for group in network.nodes:
            self._clear(group, row, p)
        for edges in network.edges.values():
            sources, targets = edges.ends(p)
            for edge in np.flatnonzero(edges.present(p) & ((sources == row) | (targets == row))):
                self._clear(edges, int(edge), p)
        hyperedges = network.hyperedges
        if hyperedges is not None and hyperedges.members is not None:
            holding = hyperedges.present(p) & (hyperedges.hyperedges(p) == row).any(axis=1)
            for hyperedge in np.flatnonzero(holding):
                self._clear(hyperedges, int(hyperedge), p)

    @staticmethod
    def _request_of(law: object) -> TopologyRequestSpec | None:
        """Return the checked topology request an update or jump law targets, or `None` for a law that targets states.

        Args:
            law:
                The law.

        Returns:
            request:
                The request.

        Raises:
            ValidationError:
                A jump changes states and requests a topology change at once, or fires a request more than once.
        """
        if isinstance(law, UpdateLaw):
            return law.target._validated() if isinstance(law.target, TopologyRequestSpec) else None
        changes = (law.changes or {}) if isinstance(law, JumpLaw) else {}
        requests = [item for item in changes if isinstance(item, TopologyRequestSpec)]
        if not requests:
            return None
        if len(changes) > 1 or changes[requests[0]] != 1:
            error_msg = (
                f'A jump requests {requests[0]!r} next to other changes or with an increment other than 1. A jump that '
                'changes the topology fires one request, `JumpLaw(rate, {request: 1})`.'
            )
            raise ValidationError(error_msg)
        return requests[0]._validated()

    @staticmethod
    def _changes_edges(request: TopologyRequestSpec) -> bool:
        """Check whether a request changes the edges of its layer: an edge request, or a new node that joins others.

        Args:
            request:
                The request.

        Returns:
            changes:
                Whether the request adds, removes, or rewires edges of its layer.

        Examples:
            >>> from skcomplex.core.specs import TopologyRequestSpec
            >>> from skcomplex.core.engine.problems._topology_kernel import TopologyKernel
            >>> rewire, removal = TopologyRequestSpec('rewire'), TopologyRequestSpec('remove-node')
            >>> TopologyKernel._changes_edges(rewire), TopologyKernel._changes_edges(removal)
            (True, False)
        """
        joins = request.edges > 0 or request.independent
        return request.kind in EDGE_KINDS or (request.kind == 'add-node' and joins)
