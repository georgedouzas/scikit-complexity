"""Evolution models on changing networks: autocatalytic sets and the extinctions of Solé and Manrubia."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import (
    Law,
    Symbols,
    Update,
    add_edge,
    add_node,
    both,
    minimum,
    mod,
    neighbors,
    normal,
    remove_node,
    total,
    uniform,
    where,
)
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System

HIDDEN = VisualSpec('sites', {'states': []})


def _initial_stream(seed: int | None) -> np.random.Generator:
    """Return a random stream of initial values that differs from the run's stream of the same seed."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


class Taxon(Element):
    """Species of an ecosystem that survives while the sum of the influences of the species it depends on is positive.

    The field `h` of the species is the sum of the couplings of its edges to the species it depends on. A species whose
    field is negative goes extinct, and a new species with `k` random dependencies takes its niche.

    Args:
        label:
            The label of the species or of the group of species.

        h:
            The field, the sum of the couplings of its dependencies.

        age:
            The number of steps since the species appeared.

        k:
            The number of dependencies of a new species.

    Examples:
        >>> from skcomplex.models.biology.evolution import Taxon
        >>> Taxon('species').k
        5
    """

    h: float = state(default=1.0, domain=Domain.REAL)
    age: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    k: int = 5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the ageing, and the extinction and replacement of a species whose field is negative.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        extinct = where(s.h < 0, 1, 0)
        return [Update(s.age, s.age + 1), Update(add_node(edges=self.k), extinct), Update(remove_node(), extinct)]


class Coupling(Element):
    """Edge from a species to a species it depends on, with the coupling `J` between them.

    A new edge, marked `fresh`, draws its coupling uniformly from `[-1, 1]` at its first step.

    Args:
        label:
            The label of the edge or of the group of edges.

        J:
            The coupling.

        fresh:
            Whether the edge has not drawn its coupling yet.

    Examples:
        >>> from skcomplex.models.biology.evolution import Coupling
        >>> Coupling('edge').fresh
        1.0
    """

    J: float = state(default=0.0, domain=Domain.REAL)
    fresh: float = state(default=1.0, domain=Domain.BOOLEAN)


class CouplingDrift(Interaction):
    """Random drift of the couplings of an ecosystem, and the first coupling of every new edge.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        couplings:
            The path of the variable group on the edges.

        noise:
            The standard deviation of the change of every coupling at a step.

    Examples:
        >>> from skcomplex.models.biology.evolution import CouplingDrift
        >>> CouplingDrift('drift', species='species', couplings='couplings').noise
        0.05
    """

    over: ClassVar[str] = 'neighbours'
    species: str = role()
    couplings: str = role()
    noise: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drift of the coupling of an edge, or its first draw.

        Args:
            s:
                The namespace of the edge, with `s.source`, `s.target`, and `s.edge`.

        Returns:
            laws:
                The updates of the coupling and of the mark of new edges.
        """
        drifted = s.edge.J + s.noise * normal('drift')
        return [
            Update(s.edge.J, where(s.edge.fresh > 1 / 2, uniform('coupling', -1, 1), drifted)),
            Update(s.edge.fresh, 0),
        ]


class DependencyField(Interaction):
    """Field of every species, the sum of the couplings of its edges to the species it depends on.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        couplings:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.biology.evolution import DependencyField
        >>> DependencyField('field', species='species', couplings='couplings').couplings
        'couplings'
    """

    species: str = role()
    couplings: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the field of every species.

        Args:
            s:
                The namespace with `s.species` and `s.couplings`.

        Returns:
            laws:
                The update of the field.
        """
        return [Update(s.species.h, neighbors(1, weight=s.couplings.J))]


class Chemical(Element):
    """Molecular species of a pool whose relative population grows with the populations of its catalysts.

    The populations follow the catalytic network: every species gains `h` times the populations of the species that
    catalyse it, and the populations are normalised, so they approach the Perron eigenvector of the network. Every
    `period` steps the least populated species, with ties broken at random, is replaced by a new species with `k`
    random catalysts, and every species catalyses the new one with probability `p`.

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

        k:
            The number of catalysts of a new species.

    Examples:
        >>> from skcomplex.models.biology.evolution import Chemical
        >>> Chemical('species').period
        10.0
    """

    x: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    tie: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    fresh: float = state(default=1.0, domain=Domain.BOOLEAN)
    h: float = parameter(default=0.5, domain=Domain.POSITIVE)
    p: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    period: float = parameter(default=10.0, domain=Domain.POSITIVE)
    k: int = 1

    def laws(self, s: Symbols) -> list[Law]:
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

        def fresh(node: Symbols) -> object:
            """Return the weight of a candidate catalysed species, 1 for the new species and 0 otherwise."""
            return node.fresh

        return [
            Update(s.x, grown / total(grown)),
            Update(s.tie, uniform('tie')),
            Update(s.fresh, 0),
            Update(add_node(edges=self.k), least),
            Update(remove_node(), least),
            Update(add_edge(toward=fresh), s.p * newcomer),
        ]


class CatalyticLink(Element):
    """Catalytic link from a species to a species that catalyses it, which carries no state of its own.

    Args:
        label:
            The label of the link or of the group of links.

    Examples:
        >>> from skcomplex.models.biology.evolution import CatalyticLink
        >>> CatalyticLink('link').label
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
            couplings `couplings.J` of the variable group `couplings` on the edges.

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
    space = NetworkSpace(graph)
    edges = graph.number_of_edges()
    couplings = rng.uniform(-1, 1, edges)
    sources, _ = space.list_edges()
    fields = np.bincount(sources, weights=couplings, minlength=n)
    species = Group(
        'species',
        element=Taxon,
        space=space,
        on='sites',
        variable=True,
        capacity=2 * n,
        values={'h': fields, 'k': k},
        visual=VisualSpec('sites', {'states': ['age']}),
    )
    edge_group = Group(
        'couplings',
        element=Coupling,
        space=space,
        on='edges',
        variable=True,
        capacity=4 * edges,
        values={'J': couplings, 'fresh': 0.0},
        visual=HIDDEN,
    )
    return System(
        components=[species, edge_group],
        interactions=[
            DependencyField('field', species='species', couplings='couplings'),
            CouplingDrift('drift', species='species', couplings='couplings', noise=noise),
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
    species = Group(
        'species',
        element=Chemical,
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
    links = Group('links', element=CatalyticLink, space=space, on='edges', variable=True, capacity=n * n, visual=HIDDEN)
    return System(components=[species, links], time=DiscreteTime())
