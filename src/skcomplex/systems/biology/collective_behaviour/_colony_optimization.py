"""Ant colonies that find and repair shortest paths on road networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet
from scipy.spatial import Delaunay

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ....core.types import Array

CUTS = 2


class TrailSegmentView(AgentView):
    """View of a trail segment, with its states as attributes.

    `TrailSegmentElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    tau: float
    passable: float


class TrailSegmentElement(Element):
    """Segment of a road network between two points.

    The segment holds the pheromone ants left on it and whether it is open.

    Args:
        label:
            The label of the segment or of the group of segments.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        tau:
            The pheromone level.

        passable:
            1 while the segment can be walked and 0 once it is cut.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import TrailSegmentElement
        >>> TrailSegmentElement('segment').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        tau: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        passable: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tau = tau
        self.passable = passable

    @classmethod
    def _get_view(cls) -> type[TrailSegmentView]:
        """Return the view of a segment, `TrailSegmentView`."""
        return TrailSegmentView


class ColonySearchInteraction(Interaction):
    """Rounds of ants that walk a road network by pheromone and length.

    The ants mark the paths they find, and the pheromone fades.

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

    draws = True

    law_classes = (RuleLaw,)

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
        """Run one round of ants and update the pheromone.

        At the round `cut`, the round also cuts the best path.

        Args:
            system:
                The system.
        """
        segments = system.get_agents(self.segments, TrailSegmentView)
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
        """Return the segments of the loop-free path of one ant to the target.

        The path has no segment if the ant does not arrive.
        """
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
        """Return the open segments after cutting two segments of the best path.

        The two segments lie in the middle of the best path, and the cut keeps the network connected.
        """
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
        """Return the neighbours of every vertex with the index of the joining segment."""
        neighbours: dict[int, list[tuple[int, int]]] = {}
        for index, (first, second) in enumerate(self.edges):
            neighbours.setdefault(first, []).append((second, index))
            neighbours.setdefault(second, []).append((first, index))
        return neighbours

    def _index(self) -> dict[frozenset[int], int]:
        """Return the index of the segment between every pair of vertices."""
        return {frozenset(pair): index for index, pair in enumerate(self.edges)}


class AntColonyOptimizationSystem(BaseSystem):
    """Ants that find the shortest path between two points of a Delaunay road network.

    The ants search round after round.

    Every round the ants walk from the point nearest one corner of the unit square towards the point nearest the
    opposite corner, choosing segments by their pheromone and length, and mark the paths they find with pheromone in
    inverse proportion to their length, while all pheromone evaporates. Short paths gather pheromone fastest, so the
    colony converges on a short path, usually the shortest, as in exercise 15.7 of Argun et al. (2021). Cutting the best
    path at the round `cut` makes the colony explore again and settle on a new route, exercise 15.12. The system has the
    pheromone `roads.tau` and the open segments `roads.passable` of the agent group `roads`, and the shortest path of
    every round `search.shortest`.

    Args:
        n_points:
            The number of points of the network.

        ants:
            The number of ants of a round.

        steps:
            The largest number of steps of an ant in a round.

        alpha:
            The exponent of the pheromone.

        beta:
            The exponent of the weight `1 / d` of a segment.

        rho:
            The evaporation rate.

        Q:
            The pheromone an ant lays along a path.

        cut:
            The round at which the best path is cut, or `None` for none.

        seed:
            The seed of the points.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import AntColonyOptimizationSystem
        >>> AntColonyOptimizationSystem(10).prepare().system_.interactions[0].ants
        20
    """

    field = 'biology/collective_behaviour'
    topics = ('ant colony optimization', 'stigmergy', 'shortest path', 'agents')
    references = ('Dorigo 1992', 'Argun et al. 2021')
    until = 100

    def __init__(
        self,
        n_points: int = 40,
        *,
        ants: int = 20,
        steps: int = 80,
        alpha: float = 0.8,
        beta: float = 1.0,
        rho: float = 0.5,
        Q: float = 1.0,
        cut: int | None = None,
        seed: int | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_points = n_points
        self.ants = ants
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.cut = cut
        self.seed = seed

    def _build_system(self) -> CompositeSystem:
        """Return the parts of ants that find the shortest path on a Delaunay road network.

        The path joins two points of the network.
        """
        points = np.random.default_rng(self.seed).random((self.n_points, 2))
        triangles = Delaunay(points).simplices
        pairs = {
            (int(min(a, b)), int(max(a, b))) for triangle in triangles for a, b in itertools.combinations(triangle, 2)
        }
        edges = sorted(pairs)
        lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in edges]
        start = int(np.argmin(np.linalg.norm(points, axis=1)))
        target = int(np.argmin(np.linalg.norm(points - 1.0, axis=1)))
        roads = GroupElement('roads', element=TrailSegmentElement, size=len(edges))
        search = ColonySearchInteraction(
            'search',
            segments='roads',
            edges=tuple(edges),
            lengths=tuple(lengths),
            start=start,
            target=target,
            ants=self.ants,
            steps=self.steps,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            Q=self.Q,
            cut=self.cut,
        )
        return CompositeSystem(self.label, elements=[roads], interactions=[search], time=DiscreteTime())
