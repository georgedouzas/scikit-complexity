"""Apply the topology requests of laws to the capacity rows of the groups of a network between the steps of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from ...core.errors import ValidationError
from ...core.laws import JumpLaw, UpdateLaw
from ...core.specs import EDGE_KINDS, TopologyRequestSpec
from ...core.types import Vector
from ._base import BaseKernel
from ._capacity_rows import CapacityRows
from ._compiled_request import CompiledRequest
from ._network_rows import NetworkRows, Pair
from ._network_snapshot import NetworkSnapshot
from ._reduction_schedule import ReductionSchedule
from ._state import State

_NO_REDUCTIONS = ReductionSchedule()


class TopologyKernel(BaseKernel):
    """Topology requests of a problem, applied between steps to the rows of the variable groups of its networks.

    `run(state)` fires the requests of update laws at the state and returns the state after them; the solver of jump
    laws applies the request of every jump channel that fires.

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

    def __init__(
        self,
        networks: tuple[NetworkRows, ...],
        requests: tuple[CompiledRequest, ...] = (),
        channels: Mapping[int, CompiledRequest] | None = None,
        reductions: ReductionSchedule = _NO_REDUCTIONS,
    ) -> None:
        self.networks = networks
        self.requests = requests
        self.channels = channels
        self.reductions = reductions

    @property
    def layers(self) -> tuple[tuple[str, str | None], ...]:
        """Path of the first group on the nodes and the layer of every network layer with a variable edge group."""
        return tuple((network.nodes[0].path, layer) for network in self.networks for layer in network.edges)

    def _decide(self, t: float, x: Vector, p: Vector, rng: np.random.Generator) -> list[tuple[int, int]]:
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
        q = self.reductions.extend(t, x, p) if any(item.reads_reductions for item in conditions) else p
        fired = []
        for index, item in enumerate(self.requests):
            assert item.condition is not None  # nosec B101: update requests carry their condition
            _, _, values = item.condition.run(State(t, x, q))
            chances = np.clip(np.asarray(values[0], dtype=float), 0.0, 1.0)
            rows = np.flatnonzero(rng.random(chances.size) < chances)
            fired += [(index, int(row)) for row in rows]
        return fired

    def _run(self, inputs: State) -> Vector:
        """Fire the requests of update laws at the state, apply them, and return the state after them.

        The requests decide at the state which rows fire, and the fired requests change a copy of the state vector and
        the parameter vector of the state in place.
        """
        t, x, p, rng = inputs.t, inputs.x, inputs.p, cast('np.random.Generator', inputs.rng)
        return self._apply(self._decide(t, x, p, rng), t, x, p, rng)

    def _apply(self, fired: list[tuple[int, int]], t: float, x: Vector, p: Vector, rng: np.random.Generator) -> Vector:
        """Apply fired requests of update laws to a copy of a state and to the parameters in place."""
        state = x.copy()
        for index, row in fired:
            self._execute(self.requests[index], row, t, state, p, rng=rng)
        return state

    def _fire(self, channel: int, row: int, t: float, x: Vector, p: Vector, *, rng: np.random.Generator) -> Vector:
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
        _, _, values = item.toward.run(State(t, x, self.reductions.extend(t, x, p)))
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
            self._add_edge(network, request.layer, (end, other), x, p, values)

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
            node = self._add_node(network, x, p, dict(request.values))
            self._join(item, node, t, x, p, rng=rng)
        elif kind == 'add-edge':
            self._join(item, row, t, x, p, rng=rng)
        elif kind == 'remove-node':
            self._remove_node(network, row, p)
        elif kind == 'remove-edge':
            self._clear(network.edges[request.layer], row, p)
        else:
            self._rewire(item, row, t, x, p, rng=rng)

    @staticmethod
    def _write(group: CapacityRows, row: int, x: Vector, p: Vector, values: Mapping[str, object]) -> None:
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
    def _clear(group: CapacityRows, row: int, p: Vector) -> None:
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

    def _add_node(self, network: NetworkRows, x: Vector, p: Vector, values: Mapping[str, object]) -> int:
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

    def _add_edge(  # noqa: PLR0917
        self,
        network: NetworkRows,
        layer: str | None,
        ends: Pair,
        x: Vector,
        p: Vector,
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
        p[edges.source.start + row] = first  # type: ignore[union-attr]
        p[edges.target.start + row] = second  # type: ignore[union-attr]

    def _remove_node(self, network: NetworkRows, row: int, p: Vector) -> None:
        """Remove a node and every edge that joins it.

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
            return law.target.check() if isinstance(law.target, TopologyRequestSpec) else None
        changes = (law.changes or {}) if isinstance(law, JumpLaw) else {}
        requests = [item for item in changes if isinstance(item, TopologyRequestSpec)]
        if not requests:
            return None
        if len(changes) > 1 or float(changes[requests[0]]) != 1:
            error_msg = (
                f'A jump requests {requests[0]!r} next to other changes or with an increment other than 1. A jump that '
                'changes the topology fires one request, `JumpLaw(rate, {request: 1})`.'
            )
            raise ValidationError(error_msg)
        return requests[0].check()

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
            >>> from skcomplex.engine.problems._topology_kernel import TopologyKernel
            >>> rewire, removal = TopologyRequestSpec('rewire'), TopologyRequestSpec('remove-node')
            >>> TopologyKernel._changes_edges(rewire), TopologyKernel._changes_edges(removal)
            (True, False)
        """
        return request.kind in EDGE_KINDS or (request.kind == 'add-node' and request.edges > 0)
