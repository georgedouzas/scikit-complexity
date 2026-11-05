"""Nonlinear consensus on temporal hypergraphs: opinions pulled by triangles that switch on and off in time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from types import SimpleNamespace

import networkx as nx
import numpy as np

from ....core.elements import GroupElement, LatticeElement
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, VisualSpec
from ..._catalog import register_model
from ._opinion import OpinionElement
from ._schedule import ScheduleElement
from ._three_body_consensus import TRIANGLE, ThreeBodyConsensusInteraction


def _spread(s: SimpleNamespace) -> object:
    """Return the squared distance of an opinion from zero, which the variance of the opinions averages."""
    return s.x**2


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'consensus', 'hypergraph', 'temporal network'),
    references=('Neuhäuser, Mellor, and Lambiotte 2020', 'Majhi et al. 2022'),
    defaults={'until': 10.0, 'dt': 0.1},
    observables=lambda system: [
        ObservableSpec('mean_opinion', group='nodes', variable='x'),
        ObservableSpec('spread', _spread, group='nodes'),
    ],
)
def hypergraph_consensus(
    n_nodes: int = 30,
    *,
    triangles: float = 0.05,
    reinforcement: float = -1.0,
    period: float = 2.0,
    seed: int = 0,
) -> System:
    """Return the nonlinear three-body consensus model on a random hypergraph of triangles that switch on and off.

    Every triple of nodes is a triangle with probability `triangles`, and every triangle is active in alternate half
    periods from a random phase. On an active triangle `ijk`, node `i` follows `x_i' = s(|x_j - x_k|) (x_j + x_k - 2
    x_i)` with `s(d) = exp(reinforcement * d)`. Without reinforcement the pulls of a triangle on its three members sum
    to zero, so the mean opinion stays fixed whatever the schedules. With reinforcement the pull depends on the member
    it acts on, and the mean opinion drifts towards the pairs that agree, for a negative reinforcement, while the
    opinions still reach a consensus on a connected hypergraph.

    Args:
        n_nodes:
            The number of nodes.

        triangles:
            The probability that a triple of nodes is a triangle.

        reinforcement:
            The exponent of the modulating function `s`.

        period:
            The period of the schedules of the triangles.

        seed:
            The seed of the hypergraph, the phases, and the opinions.

    Returns:
        system:
            The model, with the opinions `nodes.x`, the mean opinion `mean_opinion`, and the mean squared opinion
            `spread`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import hypergraph_consensus
        >>> system = hypergraph_consensus(8, triangles=0.5)
        >>> system.elements[0].prepare().size_
        8
    """
    rng = np.random.default_rng(seed)
    triads = [triad for triad in itertools.combinations(range(n_nodes), TRIANGLE) if rng.random() < triangles]
    space = NetworkSpace(nx.complete_graph(n_nodes), hyperedges=triads)
    nodes = LatticeElement(
        'nodes',
        element=OpinionElement,
        space=space,
        values={'x': rng.uniform(-1.0, 1.0, n_nodes)},
        visual=VisualSpec('sites', {'states': ['x'], 'low': -1.0, 'high': 1.0}),
    )
    schedules = GroupElement(
        'triangles',
        element=ScheduleElement,
        space=space,
        on='hyperedges',
        values={'phase': rng.uniform(0.0, 2 * math.pi, len(triads))},
    )
    consensus = ThreeBodyConsensusInteraction(
        'consensus',
        target='nodes',
        triangles='triangles',
        reinforcement=reinforcement,
        period=period,
    )
    return System(elements=[nodes, schedules], interactions=[consensus])
