"""Evolution models on changing networks: autocatalytic sets and the extinctions of Solé and Manrubia."""

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
from ._catalytic_link import CatalyticLinkElement
from ._chemical import ChemicalElement
from ._dependency_field import DependencyFieldInteraction
from ._taxon import TaxonElement

HIDDEN = VisualSpec('sites', {'states': []})


@register_model(
    field='biology/evolution',
    tags=('adaptive network', 'extinction', 'avalanches', 'coevolution', 'stochastic map'),
    references=('Solé and Manrubia 1996', 'Thurner et al. 2018'),
    defaults={'until': 300, 'dt': 1.0},
)
def sole_manrubia(
    n: int = 100,
    k: int = 5,
    *,
    noise: float = 0.05,
    seed: int | None = 1,
) -> System:
    """Return the ecosystem of Solé and Manrubia, whose species go extinct when their dependencies turn against them.

    Every species depends on `k` species through couplings drawn from `[-1, 1]`, and its field is the sum of these
    couplings. The couplings drift at random. A species whose field turns negative goes extinct, the species that
    depend on it lose a coupling, and their fields may turn negative in turn, so extinctions come in avalanches with a
    broad distribution of sizes. A new species with `k` random dependencies replaces every extinct one.

    Args:
        n:
            The number of species.

        k:
            The number of dependencies of every species.

        noise:
            The standard deviation of the drift of every coupling at a step.

        seed:
            The seed of the network and of the initial couplings.

    Returns:
        system:
            The model, with the fields `species.h` and ages `species.age` of the variable group `species` and the
            couplings `couplings.w` of the variable group `couplings` on the edges.

    Examples:
        >>> from skcomplex.models.biology.evolution import sole_manrubia
        >>> [component.label for component in sole_manrubia(n=20).elements]
        ['species', 'couplings']
    """
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for node in range(n):
        others = rng.choice(np.delete(np.arange(n), node), size=k, replace=False)
        graph.add_edges_from((node, int(other)) for other in others)
    space = NetworkSpace(graph).prepare()
    edges = graph.number_of_edges()
    couplings = rng.uniform(-1, 1, edges)
    sources, _ = space.edges_[None]
    fields = np.bincount(sources, weights=couplings, minlength=n)
    species = GroupElement(
        'species',
        element=TaxonElement,
        space=space,
        on='sites',
        variable=True,
        capacity=2 * n,
        values={'h': fields},
        visual=VisualSpec('sites', {'states': ['age']}),
    )
    edge_group = GroupElement(
        'couplings',
        element=WeightedEdgeElement,
        space=space,
        on='edges',
        variable=True,
        capacity=4 * edges,
        values={'w': couplings, 'fresh': 0.0},
        visual=HIDDEN,
    )
    return System(
        elements=[species, edge_group],
        interactions=[
            DependencyFieldInteraction('field', species='species', couplings='couplings'),
            EdgeDrawInteraction('drift', target='species', edges='couplings', noise=noise),
            MapInteraction('ageing', target='species', state='age', function=lambda s: 1, mode='increment'),
            TopologyInteraction(
                'extinction',
                target='species',
                action='replace-node',
                condition=lambda s: s.source.h < 0,
                edges=k,
            ),
        ],
        time=DiscreteTime(),
    )


@register_model(
    field='biology/evolution',
    tags=('adaptive network', 'autocatalytic set', 'Perron eigenvalue', 'origin of life', 'stochastic map'),
    references=('Jain and Krishna 1998', 'Jain and Krishna 2001', 'Thurner et al. 2018'),
    defaults={'until': 1500, 'dt': 5.0},
)
def jain_krishna(
    n: int = 40,
    *,
    p: float = 0.02,
    h: float = 0.5,
    period: int = 10,
    seed: int | None = 1,
) -> System:
    """Return the model of Jain and Krishna, where selection of the fittest builds autocatalytic sets of molecules.

    The populations of `n` molecular species approach the Perron eigenvector of their catalytic network between graph
    updates, every `period` steps. At every graph update the least populated species is replaced by a new one, linked
    to the others with probability `p`. Once a cycle of catalysts forms by chance, the species it supports win, and the
    autocatalytic set grows until it spans the network, so the number of links and the populated species jump.

    Args:
        n:
            The number of species.

        p:
            The probability of every catalytic link.

        h:
            The growth of a step from the catalysts.

        period:
            The number of steps between two graph updates.

        seed:
            The seed of the initial network.

    Returns:
        system:
            The model, with the populations `species.x` of the variable group `species` on the nodes and the variable
            group `links` on the edges, where an edge from `i` to `j` means that `j` catalyses `i`.

    Examples:
        >>> from skcomplex.models.biology.evolution import jain_krishna
        >>> round(float(jain_krishna(n=10).elements[0].values['x'].sum()), 6)
        1.0
    """
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(n, p, seed=int(rng.integers(2**31)), directed=True)
    space = NetworkSpace(graph)
    species = GroupElement(
        'species',
        element=ChemicalElement,
        space=space,
        on='sites',
        variable=True,
        capacity=2 * n,
        values={
            'x': np.full(n, 1 / n),
            'tie': rng.random(n),
            'fresh': 0.0,
            'h': h,
            'p': p,
            'period': float(period),
            'k': max(1, round(p * n)),
        },
        visual=VisualSpec('sites', {'states': ['x']}),
    )
    links = GroupElement(
        'links',
        element=CatalyticLinkElement,
        space=space,
        on='edges',
        variable=True,
        capacity=n * n,
        visual=HIDDEN,
    )
    return System(elements=[species, links], time=DiscreteTime())
