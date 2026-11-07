"""Collective behaviour systems: ant colonies that find and repair shortest paths on road networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

import numpy as np
from scipy.spatial import Delaunay

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._colony_search import ColonySearchInteraction
from ._trail_segment import TrailSegmentElement


@register_system(
    field='biology/collective_behaviour',
    tags=('ant colony optimization', 'stigmergy', 'shortest path', 'agents'),
    references=('Dorigo 1992', 'Argun et al. 2021'),
    defaults={'until': 100},
)
def ant_colony_optimization(
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
) -> System:
    """Return ants that find the shortest path between two points of a Delaunay road network, round after round.

    Every round the ants walk from the point nearest one corner of the unit square towards the point nearest the
    opposite corner, choosing segments by their pheromone and length, and mark the paths they find with pheromone in
    inverse proportion to their length, while all pheromone evaporates. Short paths gather pheromone fastest, so the
    colony converges on a short path, usually the shortest, as in exercise 15.7 of Argun et al. (2021). Cutting the
    best path at the round `cut` makes the colony explore again and settle on a new route, exercise 15.12.

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

    Returns:
        system:
            The system, with the pheromone `roads.tau` and the open segments `roads.passable` of the agent group
            `roads`, and the shortest path of every round `search.shortest`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ant_colony_optimization
        >>> ant_colony_optimization(10).interactions[0].ants
        20
    """
    points = np.random.default_rng(seed).random((n_points, 2))
    triangles = Delaunay(points).simplices
    pairs = {(int(min(a, b)), int(max(a, b))) for triangle in triangles for a, b in itertools.combinations(triangle, 2)}
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
        ants=ants,
        steps=steps,
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=Q,
        cut=cut,
    )
    return System(elements=[roads], interactions=[search], time=DiscreteTime())
