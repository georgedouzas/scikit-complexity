"""Apply the topology requests of laws to the capacity rows of the groups of a network between the steps of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...core.laws import JumpLaw, UpdateLaw
from ...errors import ValidationError
from ...specs import EDGE_KINDS, TopologyRequestSpec
from ...types import Array
from ..backends import BaseBackend
from ..problems import NetworkSnapshot, ReductionSchedule
from ._compiled_request import CompiledRequest
from ._network_rows import NetworkRows


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
    reductions: ReductionSchedule = field(default_factory=ReductionSchedule)

    @property
    def layers(self) -> tuple[tuple[str, str | None], ...]:
        """Path of the first group on the nodes and the layer of every network layer with a variable edge group."""
        return tuple((network.nodes[0].path, layer) for network in self.networks for layer in network.edges)

    def decide(self, t: float, x: Array, p: Array, rng: np.random.Generator) -> list[tuple[int, int]]:
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
        q = self.reductions.extend(t, x, p) if BaseBackend._reads_reductions(conditions) else p
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
        x: Array,
        p: Array,
        rng: np.random.Generator,
    ) -> Array:
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

    def fire(self, channel: int, row: int, t: float, x: Array, p: Array, *, rng: np.random.Generator) -> Array:
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

    def read_network(self, p: Array, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshot:
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
            for group in (*network.nodes, *network.edges.values()):
                if any(entry.name == name for entry, _ in group.variables.values()):
                    return group.present(p)
        return None

    def _weights(self, item: CompiledRequest, t: float, x: Array, p: Array) -> Array:
        """Return the weight of every node row as a new end of a request."""
        size = item.network.nodes[0].active.rows
        if item.toward is None:
            return np.ones(size)
        _, _, values = item.toward.evaluate(t, x, self.reductions.extend(t, x, p))
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
    def _allowed(item: CompiledRequest, end: int, p: Array) -> npt.NDArray[np.bool_]:
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
        x: Array,
        p: Array,
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
        x: Array,
        p: Array,
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
            >>> from skcomplex.specs import TopologyRequestSpec
            >>> from skcomplex.engine.compile._topology import Topology
            >>> rewire, removal = TopologyRequestSpec('rewire'), TopologyRequestSpec('remove-node')
            >>> Topology._changes_edges(rewire), Topology._changes_edges(removal)
            (True, False)
        """
        return request.kind in EDGE_KINDS or (request.kind == 'add-node' and request.edges > 0)
