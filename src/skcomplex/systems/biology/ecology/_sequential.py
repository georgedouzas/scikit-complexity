"""A sequentially linear ecosystem.

It is linear between extinctions that rewrite its network.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors, total, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement


class AbundanceElement(Element):
    """Species of an ecosystem with its relative abundance and its age.

    It goes extinct below a threshold.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The relative abundance.

        age:
            The number of steps since the species appeared.

        threshold:
            The abundance below which the species goes extinct.

        space:
            The network whose nodes hold one species each, or `None` for one species.

    Examples:
        >>> from skcomplex.systems.biology.ecology import AbundanceElement
        >>> AbundanceElement('species').threshold
        0.001
    """

    visual = VisualSpec('sites', {'states': ['x']})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.01,
        age: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        threshold: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.001,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.age = age
        self.threshold = threshold


class InfluenceEdgeElement(WeightedEdgeElement):
    """Influence of one species on another, hidden from the drawing."""

    visual = VisualSpec('sites', {'states': []})


class LinearGrowthInteraction(Interaction):
    """Linear growth of every species by the abundances of the species it reads.

    The abundances are normalised to one.

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
        species: Role,
        influences: Role,
        rate: Annotated[float, RealSet.open(0, oo)] = 0.2,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.influences = influences
        self.rate = rate

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the linear growth and the normalisation of the abundances.

        Args:
            context:
                The namespace with `context.species` and `context.influences`.
        """
        super()._prepare(context)
        species = context.species
        grown = species.x + context.rate * neighbors(species.other.x, weight=context.influences.w)
        kept = where(grown > 0, grown, 0)
        self.laws_: list[BaseLaw] = [UpdateLaw(species.x, kept / total(kept))]


class SequentiallyLinearEcosystemSystem(BaseSystem):
    """An ecosystem with linear dynamics between extinctions that rewrite its network.

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

        random_state:
            The seed of the network and of the initial strengths.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        random_state: int | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.k = k
        self.rate = rate
        self.threshold = threshold
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of an ecosystem.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        for node in range(self.n):
            others = rng.choice(np.delete(np.arange(self.n), node), size=self.k, replace=False)
            graph.add_edges_from((node, int(other)) for other in others)
        space = NetworkSpace(graph)
        edges = graph.number_of_edges()
        species = AbundanceElement('species', x=np.full(self.n, 1 / self.n), threshold=self.threshold, space=space)
        influences = InfluenceEdgeElement(
            'influences',
            w=rng.uniform(-1, 1, edges),
            fresh=np.zeros(edges),
            space=space,
            on='edges',
        )
        self.system_ = System(
            self.label,
            elements=[species, influences],
            interactions=[
                LinearGrowthInteraction('growth', species='species', influences='influences', rate=self.rate),
                EdgeDrawInteraction('draw', among='species'),
                Interaction(
                    'ageing',
                    roles={'target': 'species'},
                    laws=lambda s: [UpdateLaw(s.target.age, s.target.age + 1, increment=True)],
                ),
                Interaction('extinction', roles={'nodes': 'species'}, laws=self._get_extinction, along=True),
            ],
        )

    def _get_extinction(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the replacement of a species whose abundance falls below its threshold by a new species."""
        extinct = where(s.source.x < s.source.threshold, 1, 0)
        return [UpdateLaw(TopologyRequestSpec('replace-node', self.k), extinct)]
