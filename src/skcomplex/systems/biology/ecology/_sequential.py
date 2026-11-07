"""Ecology system of a sequentially linear ecosystem, linear between extinctions that rewrite its network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction, MapInteraction, TopologyInteraction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import neighbors, total, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement

HIDDEN = VisualSpec('sites', {'states': []})


class AbundanceElement(Element):
    """Species of an ecosystem with its relative abundance and its age, which goes extinct below a threshold.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The relative abundance.

        age:
            The number of steps since the species appeared.

        threshold:
            The abundance below which the species goes extinct.

    Examples:
        >>> from skcomplex.systems.biology.ecology import AbundanceElement
        >>> AbundanceElement('species').threshold
        0.001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.age = age
        self.threshold = threshold


class LinearGrowthInteraction(Interaction):
    """Linear growth of every species by the abundances of the species that influence it, normalised to one.

    Every species grows by `rate` times the sum of the abundances of the species it reads weighted by the strengths `w`
    of the influences, its abundance stays nonnegative, and the abundances are normalised to sum to one.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        influences:
            The path of the variable group on the edges.

        rate:
            The growth per step per unit of influence.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LinearGrowthInteraction
        >>> LinearGrowthInteraction('growth', species='species', influences='influences').rate
        0.2
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        influences: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.influences = influences
        self.rate = rate

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the linear growth and the normalisation of the abundances.

        Args:
            namespace:
                The namespace with `namespace.species` and `namespace.influences`.

        Returns:
            laws:
                The update of the abundance.
        """
        species = namespace.species
        grown = species.x + namespace.rate * neighbors(species.other.x, weight=namespace.influences.w)
        kept = where(grown > 0, grown, 0)
        return [UpdateLaw(species.x, kept / total(kept))]


class SequentiallyLinearEcosystemSystem(BaseSystem):
    """An ecosystem whose abundances follow linear dynamics between extinctions that rewrite its network.

    Every species reads `k` random species through influences of random sign and strength, and grows linearly with their
    abundances. Between extinctions the dynamics is linear, so the abundances approach the dominant eigenvector of the
    current network. A species whose abundance falls below `threshold` goes extinct with its influences, and a new
    species with `k` random influences takes its place, which changes the network and starts a new linear stretch.
    Extinctions come in bursts separated by quiet periods. The system has the abundances `species.x` and ages
    `species.age` of the variable group `species` and the strengths `influences.w` of the variable group `influences` on
    the edges.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import SequentiallyLinearEcosystemSystem
        >>> [component.label for component in SequentiallyLinearEcosystemSystem(n=10).prepare().system_.elements]
        ['species', 'influences']
    """

    field = 'biology/ecology'
    topics = ('adaptive network', 'extinction', 'diversity', 'linear dynamics', 'stochastic map')
    references = ('Thurner et al. 2018', 'Thurner, Klimek, and Hanel 2010')
    until = 300
    dt = 1.0

    def __init__(
        self,
        n: int = 60,
        k: int = 4,
        *,
        rate: float = 0.2,
        threshold: float = 1e-3,
        seed: int | None = 1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.k = k
        self.rate = rate
        self.threshold = threshold
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of an ecosystem."""
        rng = np.random.default_rng(self.seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        for node in range(self.n):
            others = rng.choice(np.delete(np.arange(self.n), node), size=self.k, replace=False)
            graph.add_edges_from((node, int(other)) for other in others)
        space = NetworkSpace(graph)
        edges = graph.number_of_edges()
        species = GroupElement(
            'species',
            element=AbundanceElement,
            space=space,
            on='sites',
            variable=True,
            capacity=2 * self.n,
            values={'x': np.full(self.n, 1 / self.n), 'threshold': self.threshold},
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
            self.label,
            elements=[species, influences],
            interactions=[
                LinearGrowthInteraction('growth', species='species', influences='influences', rate=self.rate),
                EdgeDrawInteraction('draw', target='species', edges='influences'),
                MapInteraction('ageing', target='species', state='age', function=lambda s: 1, mode='increment'),
                TopologyInteraction(
                    'extinction',
                    target='species',
                    action='replace-node',
                    condition=lambda s: s.source.x < s.source.threshold,
                    edges=self.k,
                ),
            ],
            time=DiscreteTime(),
        )
