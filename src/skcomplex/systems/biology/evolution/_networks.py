"""Evolution systems on changing networks.

They cover autocatalytic sets and the extinctions of Solé and Manrubia.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import both, minimum, mod, neighbors, total, uniform, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.networks import EdgeDrawInteraction, WeightedEdgeElement


class ChemicalElement(Element):
    """Molecular species of a pool whose population grows with its catalysts.

    The relative population grows with the populations of its catalysts under an `AutocatalyticGrowthInteraction`.

    Args:
        label:
            The label of the species or of the group of species.

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

        space:
            The network whose nodes hold one species each, or `None` for one species.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ChemicalElement
        >>> ChemicalElement('species').period
        10.0
    """

    visual = VisualSpec('sites', {'states': ['x']})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        tie: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.5,
        fresh: Annotated[float | Array, FiniteEnumeratedSet([0, 1])] = 1.0,
        h: Annotated[float | Array, RealSet.open(0, oo)] = 0.5,
        p: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.02,
        period: Annotated[float | Array, RealSet.open(0, oo)] = 10.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.tie = tie
        self.fresh = fresh
        self.h = h
        self.p = p
        self.period = period


class AutocatalyticGrowthInteraction(Interaction):
    """Catalytic growth of the species of a pool and the replacement of the least populated one.

    The populations follow the catalytic network: every species gains `h` times the populations of the species that
    catalyse it, and the populations are normalised, so they approach the Perron eigenvector of the network. Every
    `period` steps the least populated species, with ties broken at random, is replaced by a new species with `k`
    random catalysts, and every species catalyses the new one with probability `p`. The growth `h`, the probability
    `p`, and the period are values of the species, a `ChemicalElement`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species on the nodes.

        k:
            The number of catalysts of a new species.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AutocatalyticGrowthInteraction
        >>> AutocatalyticGrowthInteraction('growth', target='species', k=2).k
        2
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, k: int = 1) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    @staticmethod
    def _get_fresh(node: SimpleNamespace) -> object:
        """Return the weight of a candidate catalysed species, 1 for the new species and 0 otherwise."""
        return node.fresh

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the catalytic growth and the replacement of the least populated species at every graph update.

        Args:
            context:
                The namespace of the group of species.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s = context.target
        grown = s.x + s.h * neighbors(s.other.x) + 1e-6
        key = s.x + 1e-9 * s.tie
        update = mod(context.t, s.period) > s.period - 3 / 2
        least = where(both(key <= minimum(key), update), 1, 0)
        newcomer = where(total(s.fresh) > 1 / 2, 1, 0) * (1 - s.fresh)
        self.laws_: list[BaseLaw] = [
            UpdateLaw(s.x, grown / total(grown)),
            UpdateLaw(s.tie, uniform('tie')),
            UpdateLaw(s.fresh, 0),
            UpdateLaw(TopologyRequestSpec('add-node', edges=self.k), least),
            UpdateLaw(TopologyRequestSpec('remove-node'), least),
            UpdateLaw(TopologyRequestSpec('add-edge', toward=self._get_fresh), s.p * newcomer),
        ]


class TaxonElement(Element):
    """Species of an ecosystem with its field and its age.

    The field is the sum of the couplings of the species it depends on.

    A species whose field is negative goes extinct, and a new species takes its niche.

    Args:
        label:
            The label of the species or of the group of species.

        h:
            The field, the sum of the couplings of its dependencies.

        age:
            The number of steps since the species appeared.

        space:
            The network whose nodes hold one species each, or `None` for one species.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TaxonElement
        >>> TaxonElement('species').h
        1.0
    """

    visual = VisualSpec('sites', {'states': ['age']})

    def __init__(
        self,
        label: str,
        *,
        h: float | Array = 1.0,
        age: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.h = h
        self.age = age


class CouplingEdgeElement(WeightedEdgeElement):
    """Coupling of one species to another, hidden from the drawing."""

    visual = VisualSpec('sites', {'states': []})


class CatalyticLinkElement(Element):
    """Catalytic link from one species to another, hidden from the drawing."""

    visual = VisualSpec('sites', {'states': []})


class CompetingSpeciesView(AgentViewProtocol, Protocol):
    """View of a competing species, with its states as attributes.

    The rules of `CompetingSpeciesElement` read them with checked types.
    """

    x: float
    links: Array


class CompetingSpeciesElement(Element):
    """Species of the competitive Jain-Krishna model.

    It holds a relative population and the signed effects of all species on it.

    The links are the row of the interaction matrix of the species, `M_ij` for every species `j`, between -1 and 1, with
    the degradation `M_ii` between -1 and 0 on the diagonal.

    Args:
        label:
            The label of the species or of the group of species.

        x:
            The relative population.

        links:
            The effects of every species on this one, one row of the interaction matrix per member.

        size:
            The number of species, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CompetingSpeciesElement
        >>> CompetingSpeciesElement('species', x=0.1).x
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        links: float | Array = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.links = links


class CompetitiveSelectionInteraction(Interaction):
    """Population dynamics of signed catalytic interactions.

    The least populated species is replaced at every step.

    Between replacements the relative populations follow Eq. (5.37) of Thurner et al. (2018), `dx_i/dt = f_i` with
    `f_i = sum_j M_ij x_j - x_i phi`, where `phi = sum_jk M_jk x_k` keeps the total at one, except that a species at
    zero stays there while `f_i < 0`. The rule integrates them for the time `duration` by Euler steps of `step`, then
    replaces the least populated species, ties broken at random, by a new one whose links to and from every other
    species are drawn from `[-1, 1]` with the probability `p`, whose degradation is drawn from `[-1, 0]`, and whose
    population starts at `1 / N`.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species.

        p:
            The probability of every link of a new species.

        duration:
            The time of the population dynamics between replacements.

        step:
            The Euler step of the population dynamics.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CompetitiveSelectionInteraction
        >>> CompetitiveSelectionInteraction('selection', species='species', p=0.1).p
        0.1
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        species: Role,
        p: float = 0.05,
        duration: float = 20.0,
        step: float = 0.05,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.p = p
        self.duration = duration
        self.step = step

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the selection, a rule of the whole system.

        Args:
            context:
                The namespace of the system.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context, self.select)]

    def select(self, system: SystemView) -> None:
        """Run the population dynamics, then replace the least populated species.

        Args:
            system:
                The system.
        """
        species = cast(list[CompetingSpeciesView], system.get_agents(self.species))
        matrix = np.array([np.asarray(item.links, dtype=float) for item in species])
        populations = self._relax(matrix, np.array([item.x for item in species]))
        size = len(species)
        weakest = np.flatnonzero(populations <= populations.min() + 1e-12)
        k = int(system.rng.choice(weakest))
        drawn = system.rng.uniform(-1, 1, (2, size)) * (system.rng.random((2, size)) < self.p)
        matrix[k, :], matrix[:, k] = drawn[0], drawn[1]
        matrix[k, k] = -system.rng.random()
        populations[k] = 1 / size
        populations /= populations.sum()
        for index, item in enumerate(species):
            item.x = float(populations[index])
            item.links = matrix[index]

    def _relax(self, matrix: Array, populations: Array) -> Array:
        """Return the populations after the dynamics of Eq.

        (5.37) for the time `duration`.
        """
        x = populations.copy()
        for _ in range(round(self.duration / self.step)):
            drive = matrix @ x
            f = drive - x * drive.sum()
            f[(x <= 0) & (f < 0)] = 0.0
            x = np.clip(x + self.step * f, 0.0, None)
            total = x.sum()
            x = x / total if total > 0 else np.full_like(x, 1 / x.size)
        return x


class DependencyFieldInteraction(Interaction):
    """Field of every species.

    The field is the sum of the couplings `w` of its edges to the species it depends on.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        couplings:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.systems.biology.evolution import DependencyFieldInteraction
        >>> DependencyFieldInteraction('field', species='species', couplings='couplings').couplings
        'couplings'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        species: Role,
        couplings: Role,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.couplings = couplings

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the field of every species.

        Args:
            context:
                The namespace with `context.species` and `context.couplings`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.species.h, neighbors(1, weight=context.couplings.w))]


class SoleManrubiaSystem(BaseSystem):
    """The ecosystem of Solé and Manrubia.

    Its species go extinct when their dependencies turn against them.

    Every species depends on `k` species through couplings drawn from `[-1, 1]`, and its field is the sum of these
    couplings. The couplings drift at random. A species whose field turns negative goes extinct, the species that depend
    on it lose a coupling, and their fields may turn negative in turn, so extinctions come in avalanches with a broad
    distribution of sizes. A new species with `k` random dependencies replaces every extinct one. The system has the
    fields `species.h` and ages `species.age` of the variable group `species` and the couplings `couplings.w` of the
    variable group `couplings` on the edges.

    Args:
        n:
            The number of species.

        k:
            The number of dependencies of every species.

        noise:
            The standard deviation of the drift of every coupling at a step.

        seed:
            The seed of the network and of the initial couplings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import SoleManrubiaSystem
        >>> [component.label for component in SoleManrubiaSystem(n=20).prepare().system_.elements]
        ['species', 'couplings']
    """

    field = 'biology/evolution'
    topics = ('adaptive network', 'extinction', 'avalanches', 'coevolution', 'stochastic map')
    references = ('Solé and Manrubia 1996', 'Thurner et al. 2018')
    until = 300
    dt = 1.0

    def __init__(
        self,
        n: int = 100,
        k: int = 5,
        *,
        noise: float = 0.05,
        seed: int | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.k = k
        self.noise = noise
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the ecosystem of Solé and Manrubia.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n))
        for node in range(self.n):
            others = rng.choice(np.delete(np.arange(self.n), node), size=self.k, replace=False)
            graph.add_edges_from((node, int(other)) for other in others)
        space = NetworkSpace(graph).prepare()
        edges = graph.number_of_edges()
        couplings = rng.uniform(-1, 1, edges)
        sources, _ = space.edges_[None]
        fields = np.bincount(sources, weights=couplings, minlength=self.n)
        species = TaxonElement('species', h=fields, space=space)
        edge_group = CouplingEdgeElement('couplings', w=couplings, fresh=np.zeros(edges), space=space, on='edges')
        self.system_ = System(
            self.label,
            elements=[species, edge_group],
            interactions=[
                DependencyFieldInteraction('field', species='species', couplings='couplings'),
                EdgeDrawInteraction('drift', among='species', noise=self.noise),
                Interaction(
                    'ageing',
                    roles={'target': 'species'},
                    laws=lambda s: [UpdateLaw(s.target.age, s.target.age + 1, increment=True)],
                ),
                Interaction('extinction', roles={'nodes': 'species'}, laws=self._get_extinction, along=True),
            ],
        )

    def _get_extinction(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the replacement of a species whose field turns negative by a new species."""
        return [UpdateLaw(TopologyRequestSpec('replace-node', self.k), where(s.source.h < 0, 1, 0))]


class JainKrishnaSystem(BaseSystem):
    """The system of Jain and Krishna.

    Selection of the fittest builds autocatalytic sets of molecules.

    The populations of `n` molecular species approach the Perron eigenvector of their catalytic network between graph
    updates, every `period` steps. At every graph update the least populated species is replaced by a new one, linked to
    the others with probability `p`. Once a cycle of catalysts forms by chance, the species it supports win, and the
    autocatalytic set grows until it spans the network, so the number of links and the populated species jump. The
    system has the populations `species.x` of the variable group `species` on the nodes and the variable group `links`
    on the edges, where an edge from `i` to `j` means that `j` catalyses `i`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import JainKrishnaSystem
        >>> round(float(JainKrishnaSystem(n=10).prepare().system_.elements[0].x.sum()), 6)
        1.0
    """

    field = 'biology/evolution'
    topics = ('adaptive network', 'autocatalytic set', 'Perron eigenvalue', 'origin of life', 'stochastic map')
    references = ('Jain and Krishna 1998', 'Jain and Krishna 2001', 'Thurner et al. 2018')
    until = 1500
    dt = 5.0

    def __init__(
        self,
        n: int = 40,
        *,
        p: float = 0.02,
        h: float = 0.5,
        period: int = 10,
        seed: int | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.p = p
        self.h = h
        self.period = period
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Jain and Krishna.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        graph = nx.gnp_random_graph(self.n, self.p, seed=int(rng.integers(2**31)), directed=True)
        space = NetworkSpace(graph)
        species = ChemicalElement(
            'species',
            x=np.full(self.n, 1 / self.n),
            tie=rng.random(self.n),
            fresh=np.zeros(self.n),
            h=self.h,
            p=self.p,
            period=float(self.period),
            space=space,
        )
        links = CatalyticLinkElement('links', space=space, on='edges')
        growth = AutocatalyticGrowthInteraction('growth', target='species', k=max(1, round(self.p * self.n)))
        self.system_ = System(self.label, elements=[species, links], interactions=[growth])


class CompetitiveJainKrishnaSystem(BaseSystem):
    """The Jain-Krishna model with signed interactions.

    Its surviving species come to cooperate.

    Every link of the interaction matrix is present with the probability `p` and drawn from `[-1, 1]`, and every species
    degrades at a rate drawn from `[-1, 0]`. The populations follow the linear catalytic dynamics with a floor at zero,
    and at every step the least populated species is replaced by a new random one. Selection keeps the species that
    catalyse each other, so once an autocatalytic set forms the links among the surviving species turn mostly positive,
    the emergence of cooperation, and diversity grows until a keystone species is lost. The system has the populations
    `x` and the rows `links` of the interaction matrix of the group `species`.

    Args:
        n:
            The number of species.

        p:
            The probability of every link.

        duration:
            The time of the population dynamics between replacements.

        seed:
            The seed of the initial matrix.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CompetitiveJainKrishnaSystem
        >>> CompetitiveJainKrishnaSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'biology/evolution'
    topics = ('autocatalytic set', 'competition', 'cooperation', 'punctuated equilibrium', 'agents')
    references = ('Jain and Krishna 2002', 'Thurner et al. 2018')
    until = 400

    def __init__(
        self,
        n: int = 50,
        *,
        p: float = 0.05,
        duration: float = 20.0,
        seed: int | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.p = p
        self.duration = duration
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Jain-Krishna model with signed interactions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        matrix = rng.uniform(-1, 1, (self.n, self.n)) * (rng.random((self.n, self.n)) < self.p)
        np.fill_diagonal(matrix, -rng.random(self.n))
        species = CompetingSpeciesElement('species', x=np.full(self.n, 1 / self.n), links=matrix, size=self.n)
        selection = CompetitiveSelectionInteraction('selection', species='species', p=self.p, duration=self.duration)
        self.system_ = System(self.label, elements=[species], interactions=[selection])
