"""Evolution systems on changing networks: autocatalytic sets and the extinctions of Solé and Manrubia."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement
from ._catalytic_link import CatalyticLinkElement
from ._chemical import ChemicalElement
from ._competing_species import CompetingSpeciesElement
from ._competitive_selection import CompetitiveSelectionInteraction
from ._dependency_field import DependencyFieldInteraction
from ._taxon import TaxonElement

HIDDEN = VisualSpec('sites', {'states': []})


@register_system(
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
            The system, with the fields `species.h` and ages `species.age` of the variable group `species` and the
            couplings `couplings.w` of the variable group `couplings` on the edges.

    Examples:
        >>> from skcomplex.systems.biology.evolution import sole_manrubia
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


@register_system(
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
    """Return the system of Jain and Krishna, where selection of the fittest builds autocatalytic sets of molecules.

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
            The system, with the populations `species.x` of the variable group `species` on the nodes and the variable
            group `links` on the edges, where an edge from `i` to `j` means that `j` catalyses `i`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import jain_krishna
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


@register_system(
    field='biology/evolution',
    tags=('autocatalytic set', 'competition', 'cooperation', 'punctuated equilibrium', 'agents'),
    references=('Jain and Krishna 2002', 'Thurner et al. 2018'),
    defaults={'until': 400},
)
def competitive_jain_krishna(
    n: int = 50,
    *,
    p: float = 0.05,
    duration: float = 20.0,
    seed: int | None = 1,
) -> System:
    """Return the Jain-Krishna model with signed interactions, whose surviving species come to cooperate.

    Every link of the interaction matrix is present with the probability `p` and drawn from `[-1, 1]`, and every
    species degrades at a rate drawn from `[-1, 0]`. The populations follow the linear catalytic dynamics with a floor
    at zero, and at every step the least populated species is replaced by a new random one. Selection keeps the species
    that catalyse each other, so once an autocatalytic set forms the links among the surviving species turn mostly
    positive, the emergence of cooperation, and diversity grows until a keystone species is lost.

    Args:
        n:
            The number of species.

        p:
            The probability of every link.

        duration:
            The time of the population dynamics between replacements.

        seed:
            The seed of the initial matrix.

    Returns:
        system:
            The system, with the populations `x` and the rows `links` of the interaction matrix of the group `species`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import competitive_jain_krishna
        >>> competitive_jain_krishna(10).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < p)
    np.fill_diagonal(matrix, -rng.random(n))
    species = GroupElement(
        'species',
        element=CompetingSpeciesElement,
        size=n,
        values={'x': np.full(n, 1 / n), 'links': matrix},
    )
    selection = CompetitiveSelectionInteraction('selection', species='species', p=p, duration=duration)
    return System(elements=[species], interactions=[selection], time=DiscreteTime())
