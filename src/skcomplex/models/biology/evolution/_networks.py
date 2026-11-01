"""Evolution models on changing networks: autocatalytic sets and the extinctions of Solé and Manrubia."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction, MapInteraction, TopologyInteraction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import both, minimum, mod, neighbors, total, uniform, where
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....core.times import DiscreteTime
from ....specs import ParameterSpec, RoleSpec, StateSpec, TopologyRequestSpec, VisualSpec
from ..._catalog import register_model
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement

HIDDEN = VisualSpec('sites', {'states': []})


def _initial_stream(seed: int | None) -> np.random.Generator:
    """Return a random stream of initial values that differs from the run's stream of the same seed."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


class TaxonElement(Element):
    """Species of an ecosystem with its field, the sum of the couplings of the species it depends on, and its age.

    A species whose field is negative goes extinct, and a new species takes its niche.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        h:
            The field, the sum of the couplings of its dependencies.

        age:
            The number of steps since the species appeared.

    Examples:
        >>> from skcomplex.models.biology.evolution import TaxonElement
        >>> TaxonElement('species').h
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec()] = 1.0,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.age = age


class DependencyFieldInteraction(Interaction):
    """Field of every species, the sum of the couplings `w` of its edges to the species it depends on.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        couplings:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.biology.evolution import DependencyFieldInteraction
        >>> DependencyFieldInteraction('field', species='species', couplings='couplings').couplings
        'couplings'
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        couplings: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.species = species
        self.couplings = couplings

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the field of every species.

        Args:
            s:
                The namespace with `s.species` and `s.couplings`.

        Returns:
            laws:
                The update of the field.
        """
        return [UpdateLaw(s.species.h, neighbors(1, weight=s.couplings.w))]


class ChemicalElement(Element):
    """Molecular species of a pool whose relative population grows with the populations of its catalysts.

    The populations follow the catalytic network: every species gains `h` times the populations of the species that
    catalyse it, and the populations are normalised, so they approach the Perron eigenvector of the network. Every
    `period` steps the least populated species, with ties broken at random, is replaced by a new species with `k`
    random catalysts, and every species catalyses the new one with probability `p`.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The relative population.

        tie:
            The random key that breaks ties between the least populated species.

        fresh:
            Whether the species appeared at the previous graph update.

        h:
            The growth of a step from the catalysts.

        p:
            The probability of every catalytic link of a new species.

        period:
            The number of steps between two graph updates.

        k:
            The number of catalysts of a new species.

    Examples:
        >>> from skcomplex.models.biology.evolution import ChemicalElement
        >>> ChemicalElement('species').period
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tie: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        fresh: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        k: int = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.tie = tie
        self.fresh = fresh
        self.h = h
        self.p = p
        self.period = period
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the catalytic growth, and the replacement of the least populated species at every graph update.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        grown = s.x + s.h * neighbors(s.other.x) + 1e-6
        key = s.x + 1e-9 * s.tie
        update = mod(s.t, s.period) > s.period - 3 / 2
        least = where(both(key <= minimum(key), update), 1, 0)
        newcomer = where(total(s.fresh) > 1 / 2, 1, 0) * (1 - s.fresh)

        def fresh(node: SimpleNamespace) -> object:
            """Return the weight of a candidate catalysed species, 1 for the new species and 0 otherwise."""
            return node.fresh

        return [
            UpdateLaw(s.x, grown / total(grown)),
            UpdateLaw(s.tie, uniform('tie')),
            UpdateLaw(s.fresh, 0),
            UpdateLaw(TopologyRequestSpec('add-node', edges=self.k), least),
            UpdateLaw(TopologyRequestSpec('remove-node'), least),
            UpdateLaw(TopologyRequestSpec('add-edge', toward=fresh), s.p * newcomer),
        ]


class CatalyticLinkElement(Element):
    """Catalytic link from a species to a species that catalyses it, which carries no state of its own.

    Args:
        label:
            The label of the link or of the group of links.

    Examples:
        >>> from skcomplex.models.biology.evolution import CatalyticLinkElement
        >>> CatalyticLinkElement('link').label
        'link'
    """


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
    rng = _initial_stream(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for node in range(n):
        others = rng.choice(np.delete(np.arange(n), node), size=k, replace=False)
        graph.add_edges_from((node, int(other)) for other in others)
    space = NetworkSpace(graph).check().prepare()
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
    rng = _initial_stream(seed)
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
