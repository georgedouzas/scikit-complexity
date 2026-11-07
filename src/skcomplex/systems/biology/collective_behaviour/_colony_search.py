"""Rounds of ants that walk a road network by pheromone and length, mark the paths they find, and let it fade."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ....core.types import Array
from ._trail_segment_view import TrailSegmentView

CUTS = 2


class ColonySearchInteraction(Interaction):
    """Rounds of ants that walk a road network by pheromone and length, mark the paths they find, and let it fade.

    In every round, following Argun et al. (2021, section 15.2), `ants` ants start at the vertex `start` and take up to
    `steps` steps. At vertex `i` an ant moves to the neighbour `j` with the probability `tau_ij^alpha w_ij^beta / sum_k
    tau_ik^alpha w_ik^beta`, with the weight `w_ij = 1 / d_ij` of the length `d_ij`, Eq. (15.2), and stops at the vertex
    `target`. The path of every ant that arrives is cleared of its loops, and every segment of it receives the pheromone
    `Q / L` of its length `L`, Eq. (15.3), after the pheromone of every segment has evaporated by the factor `1 - rho`,
    Eq. (15.4). At the round `cut`, two segments near the middle of the shortest path found so far are cut, as long as
    the network stays connected, and the colony must heal the disruption. The interaction reports the length of the
    shortest path of every round as `shortest`.

    Args:
        label:
            The label of the interaction.

        segments:
            The path of the group of segments.

        edges:
            The two ends of every segment, in the order of the group.

        lengths:
            The length of every segment.

        start:
            The vertex the ants start from.

        target:
            The vertex the ants look for.

        ants:
            The number of ants of a round.

        steps:
            The largest number of steps of an ant in a round.

        alpha:
            The exponent of the pheromone.

        beta:
            The exponent of the weight.

        rho:
            The evaporation rate.

        Q:
            The pheromone an ant lays along a path.

        cut:
            The round at which the shortest path found is cut, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ColonySearchInteraction
        >>> ColonySearchInteraction('search', segments='roads', edges=((0, 1),), lengths=(1.0,), target=1).ants
        20
    """

    reports: ClassVar[tuple[str, ...]] = ('shortest',)

    def __init__(
        self,
        label: str,
        *,
        segments: Annotated[str, RoleSpec()],
        edges: Sequence[tuple[int, int]] = (),
        lengths: Sequence[float] = (),
        start: int = 0,
        target: int = 1,
        ants: int = 20,
        steps: int = 80,
        alpha: float = 0.8,
        beta: float = 1.0,
        rho: float = 0.5,
        Q: float = 1.0,
        cut: int | None = None,
    ) -> None:
        super().__init__(label)
        self.segments = segments
        self.edges = edges
        self.lengths = lengths
        self.start = start
        self.target = target
        self.ants = ants
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.cut = cut

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of a round.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.search)]

    def search(self, system: SystemView) -> None:
        """Run one round of ants, cut the best path at the round `cut`, and update the pheromone.

        Args:
            system:
                The system.
        """
        segments = system.agents(self.segments, TrailSegmentView)
        tau = np.array([item.tau for item in segments])
        is_open = np.array([item.passable for item in segments]) > 0
        best = system.storage.get('best')
        if (
            self.cut is not None
            and system.time >= self.cut
            and isinstance(best, list)
            and not system.storage.get('cut')
        ):
            is_open = self._disrupt(is_open, best)
            tau[~is_open] = 0.0
            system.storage['cut'] = True
        paths = [self._walk(tau, is_open, system.rng) for _ in range(self.ants)]
        found = [path for path in paths if path]
        lengths = np.asarray(self.lengths, dtype=float)
        deposit = np.zeros_like(tau)
        shortest = np.inf
        for path in found:
            total = float(lengths[path].sum())
            deposit[path] += self.Q / total
            if total < shortest:
                shortest = total
                system.storage['best'] = path
        tau = ((1 - self.rho) * tau + deposit) * is_open
        for item, level, state in zip(segments, tau, is_open, strict=True):
            item.tau = float(level)
            item.passable = float(state)
        system.report('shortest', shortest if np.isfinite(shortest) else float('nan'))

    def _walk(self, tau: Array, is_open: npt.NDArray[np.bool_], rng: np.random.Generator) -> list[int]:
        """Return the segments of the loop-free path of one ant to the target, or no segment if it does not arrive."""
        neighbours = self._neighbours()
        lengths = np.asarray(self.lengths, dtype=float)
        vertex, visited = self.start, [self.start]
        for _ in range(self.steps):
            options = [(other, edge) for other, edge in neighbours[vertex] if is_open[edge]]
            if not options:
                return []
            edges = np.array([edge for _, edge in options])
            scores = tau[edges] ** self.alpha * (1 / lengths[edges]) ** self.beta + 1e-12
            vertex = options[int(rng.choice(len(options), p=scores / scores.sum()))][0]
            if vertex in visited:
                visited = visited[: visited.index(vertex) + 1]
            else:
                visited.append(vertex)
            if vertex == self.target:
                index = self._index()
                return [index[frozenset(pair)] for pair in itertools.pairwise(visited)]
        return []

    def _disrupt(self, is_open: npt.NDArray[np.bool_], best: list[int]) -> npt.NDArray[np.bool_]:
        """Return the open segments after cutting two segments of the middle of the best path, keeping it connected."""
        cut = is_open.copy()
        middle = len(best) // 2
        removed = 0
        for edge in best[middle:] + best[:middle]:
            if removed == CUTS:
                break
            trial = cut.copy()
            trial[edge] = False
            graph = nx.Graph([pair for pair, keep in zip(self.edges, trial, strict=True) if keep])
            if (
                graph.has_node(self.start)
                and graph.has_node(self.target)
                and nx.has_path(graph, self.start, self.target)
            ):
                cut = trial
                removed += 1
        return cut

    def _neighbours(self) -> dict[int, list[tuple[int, int]]]:
        """Return the neighbours of every vertex with the index of the segment that joins them."""
        neighbours: dict[int, list[tuple[int, int]]] = {}
        for index, (first, second) in enumerate(self.edges):
            neighbours.setdefault(first, []).append((second, index))
            neighbours.setdefault(second, []).append((first, index))
        return neighbours

    def _index(self) -> dict[frozenset[int], int]:
        """Return the index of the segment between every pair of vertices."""
        return {frozenset(pair): index for index, pair in enumerate(self.edges)}
