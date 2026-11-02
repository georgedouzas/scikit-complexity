"""Ecology model of a sequentially linear ecosystem, linear between extinctions that rewrite its interaction network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction, TopologyInteraction
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import VisualSpec
from ..._catalog import register_model
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement
from ._abundance import AbundanceElement
from ._linear_growth import LinearGrowthInteraction

HIDDEN = VisualSpec('sites', {'states': []})


@register_model(
    field='biology/ecology',
    tags=('adaptive network', 'extinction', 'diversity', 'linear dynamics', 'stochastic map'),
    references=('Thurner et al. 2018', 'Thurner, Klimek, and Hanel 2010'),
    defaults={'until': 300, 'dt': 1.0},
)
def sequentially_linear_ecosystem(
    n: int = 60,
    k: int = 4,
    *,
    rate: float = 0.2,
    threshold: float = 1e-3,
    seed: int | None = 1,
) -> System:
    """Return an ecosystem whose abundances follow linear dynamics between extinctions that rewrite its network.

    Every species reads `k` random species through influences of random sign and strength, and grows linearly with
    their abundances. Between extinctions the dynamics is linear, so the abundances approach the dominant eigenvector
    of the current network. A species whose abundance falls below `threshold` goes extinct with its influences, and a
    new species with `k` random influences takes its place, which changes the network and starts a new linear stretch.
    Extinctions come in bursts separated by quiet periods.

    Args:
        n:
            The number of species.

        k:
            The number of species every species reads.

        rate:
            The growth per step per unit of influence.

        threshold:
            The abundance below which a species goes extinct.

        seed:
            The seed of the network and of the initial strengths.

    Returns:
        system:
            The model, with the abundances `species.x` and ages `species.age` of the variable group `species` and the
            strengths `influences.w` of the variable group `influences` on the edges.

    Examples:
        >>> from skcomplex.models.biology.ecology import sequentially_linear_ecosystem
        >>> [component.label for component in sequentially_linear_ecosystem(n=10).elements]
        ['species', 'influences']
    """
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for node in range(n):
        others = rng.choice(np.delete(np.arange(n), node), size=k, replace=False)
        graph.add_edges_from((node, int(other)) for other in others)
    space = NetworkSpace(graph)
    edges = graph.number_of_edges()
    species = GroupElement(
        'species',
        element=AbundanceElement,
        space=space,
        on='sites',
        variable=True,
        capacity=2 * n,
        values={'x': np.full(n, 1 / n), 'threshold': threshold},
        visual=VisualSpec('sites', {'states': ['x']}),
    )
    influences = GroupElement(
        'influences',
        element=WeightedEdgeElement,
        space=space,
        on='edges',
        variable=True,
        capacity=4 * edges,
        values={'w': rng.uniform(-1, 1, edges), 'fresh': 0.0},
        visual=HIDDEN,
    )
    return System(
        elements=[species, influences],
        interactions=[
            LinearGrowthInteraction('growth', species='species', influences='influences', rate=rate),
            EdgeDrawInteraction('draw', target='species', edges='influences'),
            MapInteraction('ageing', target='species', state='age', function=lambda s: 1, mode='increment'),
            TopologyInteraction(
                'extinction',
                target='species',
                action='replace-node',
                condition=lambda s: s.source.x < s.source.threshold,
                edges=k,
            ),
        ],
        time=DiscreteTime(),
    )
