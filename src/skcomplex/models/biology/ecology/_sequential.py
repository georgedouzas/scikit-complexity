"""Ecology model of a sequentially linear ecosystem, linear between extinctions that rewrite its interaction network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import networkx as nx
import numpy as np

from ....core.base import BaseInteraction, BaseLaw, NumericDomain, VisualSpec, parameter, role, state
from ....core.components import Element, GroupElement
from ....core.laws import Symbols, UpdateLaw, neighbors, total, uniform, where
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....formalisms.topology import add_node, remove_node

HIDDEN = VisualSpec('sites', {'states': []})


class AbundanceElement(Element):
    """Species of an ecosystem with its relative abundance, which goes extinct below a threshold.

    A species whose abundance falls below `threshold` goes extinct, and a new species with `k` random interactions
    and the abundance `seed` takes its niche.

    Args:
        label:
            The label of the species or of the group of species.

        k:
            The number of interactions of a new species.

        **values:
            The values of the declared variables, by name:

            x:
                The relative abundance.

            age:
                The number of steps since the species appeared.

            threshold:
                The abundance below which the species goes extinct.

    Examples:
        >>> from skcomplex.models.biology.ecology import AbundanceElement
        >>> AbundanceElement('species').threshold
        0.001
    """

    x: float = state(default=0.01, domain=NumericDomain.NONNEGATIVE)
    age: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    threshold: float = parameter(default=1e-3, domain=NumericDomain.NONNEGATIVE)

    def __init__(
        self,
        label: str,
        *,
        k: int = 4,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.k = k

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the ageing, and the extinction and replacement of a species below the threshold.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        extinct = where(s.x < s.threshold, 1, 0)
        return [
            UpdateLaw(s.age, s.age + 1),
            UpdateLaw(add_node(edges=self.k), extinct),
            UpdateLaw(remove_node(), extinct),
        ]


class InfluenceElement(Element):
    """Edge from a species to a species that changes its growth, with the strength `a` of the influence.

    A new edge, marked `fresh`, draws its strength uniformly from `[-1, 1]` at its first step.

    Args:
        label:
            The label of the edge or of the group of edges.

        a:
            The strength of the influence.

        fresh:
            Whether the edge has not drawn its strength yet.

    Examples:
        >>> from skcomplex.models.biology.ecology import InfluenceElement
        >>> InfluenceElement('edge').a
        0.0
    """

    a: float = state(default=0.0, domain=NumericDomain.REAL)
    fresh: float = state(default=1.0, domain=NumericDomain.BOOLEAN)


class InfluenceDrawInteraction(BaseInteraction):
    """First draw of the strength of every new influence of an ecosystem.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        influences:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.biology.ecology import InfluenceDrawInteraction
        >>> InfluenceDrawInteraction('draw', species='species', influences='influences').influences
        'influences'
    """

    over: ClassVar[str] = 'neighbors'
    species: str = role()
    influences: str = role()

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the draw of the strength of a new edge.

        Args:
            s:
                The namespace of the edge, with `s.source`, `s.target`, and `s.edge`.

        Returns:
            laws:
                The updates of the strength and of the mark of new edges.
        """
        drawn = uniform('strength', -1, 1)
        return [UpdateLaw(s.edge.a, where(s.edge.fresh > 1 / 2, drawn, s.edge.a)), UpdateLaw(s.edge.fresh, 0)]


class LinearGrowthInteraction(BaseInteraction):
    """Linear growth of every species by the abundances of the species that influence it, normalised to one.

    Every species grows by `rate` times the sum of the abundances of the species it reads weighted by the strengths of
    the influences, its abundance stays nonnegative, and the abundances are normalised to sum to one.

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
        >>> from skcomplex.models.biology.ecology import LinearGrowthInteraction
        >>> LinearGrowthInteraction('growth', species='species', influences='influences').rate
        0.2
    """

    species: str = role()
    influences: str = role()
    rate: float = parameter(default=0.2, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the linear growth and the normalisation of the abundances.

        Args:
            s:
                The namespace with `s.species` and `s.influences`.

        Returns:
            laws:
                The update of the abundance.
        """
        species = s.species
        grown = species.x + s.rate * neighbors(species.other.x, weight=s.influences.a)
        kept = where(grown > 0, grown, 0)
        return [UpdateLaw(species.x, kept / total(kept))]


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
    new species with random influences takes its place, which changes the network and starts a new linear stretch.
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
            strengths `influences.a` of the variable group `influences` on the edges.

    Examples:
        >>> from skcomplex.models.biology.ecology import sequentially_linear_ecosystem
        >>> [component.label for component in sequentially_linear_ecosystem(n=10).elements]
        ['species', 'influences']
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
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
        values={'x': np.full(n, 1 / n), 'threshold': threshold, 'k': k},
        visual=VisualSpec('sites', {'states': ['x']}),
    )
    influences = GroupElement(
        'influences',
        element=InfluenceElement,
        space=space,
        on='edges',
        variable=True,
        capacity=4 * edges,
        values={'a': rng.uniform(-1, 1, edges), 'fresh': 0.0},
        visual=HIDDEN,
    )
    return System(
        components=[species, influences],
        interactions=[
            LinearGrowthInteraction('growth', species='species', influences='influences', rate=rate),
            InfluenceDrawInteraction('draw', species='species', influences='influences'),
        ],
        time=DiscreteTime(),
    )
