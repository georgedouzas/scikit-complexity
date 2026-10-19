"""Epidemiology models: infection on networks in the mean-field limit and between subpopulations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, state, vector
from ....core.model import Element, EuclideanSpace, Field, Law, NetworkSpace, Rate, Symbols, System, laplacian
from ....core.runtime import register_model


class DegreeClasses(Element):
    """Infected densities of the classes of nodes of equal degree of a network, in heterogeneous mean field.

    A node of degree `k` recovers at rate 1 and is infected at rate `lam k Theta`, where
    `Theta = sum_k k P(k) rho_k / <k>` is the probability that an edge points to an infected node, so
    `d rho_k / dt = -rho_k + lam k (1 - rho_k) Theta`. The densities are one vector component per class.

    Args:
        label:
            The label of the element.

        rho:
            The initial infected density of every class.

        lam:
            The spreading rate, the ratio of the infection and recovery rates.

        degrees:
            The degree of every class.

        weights:
            The probability `P(k)` of every class.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import DegreeClasses
        >>> DegreeClasses('network', degrees=(1, 2), weights=(0.5, 0.5)).degrees
        (1, 2)
    """

    rho: npt.ArrayLike = state(default=0.01, domain=Domain.NONNEGATIVE, vector=True)
    lam: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    degrees: Sequence[float] = (1.0,)
    weights: Sequence[float] = (1.0,)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the recovery and the infection along edges of every class.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        degrees, weights = np.asarray(self.degrees, dtype=float), np.asarray(self.weights, dtype=float)
        mean = float(degrees @ weights)
        theta = sum(degree * weight * density for degree, weight, density in zip(degrees, weights, s.rho, strict=True))
        infection = vector([s.lam * degree * (1 - density) for degree, density in zip(degrees, s.rho, strict=True)])
        return [Rate(s.rho, -s.rho + infection * theta / mean)]


class Subpopulation(Element):
    """Well-mixed SIR population at a node of a network, whose individuals hop along each edge at the rate `D`.

    Inside the node, `beta S I / N` individuals are infected and `gamma I` recover per unit time, with
    `N = S + I + R`. Hopping along every edge adds `D` times the graph Laplacian to every compartment.

    Args:
        label:
            The label of the subpopulation or the field.

        S:
            The initial number of susceptible individuals.

        I:
            The initial number of infected individuals.

        R:
            The initial number of recovered individuals.

        beta:
            The infection rate.

        gamma:
            The recovery rate.

        D:
            The rate at which an individual hops along one edge.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Subpopulation
        >>> Subpopulation('city', S=1000.0).beta
        0.3
    """

    S: float = state(default=1000.0, domain=Domain.NONNEGATIVE)
    I: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    D: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the infections, recoveries, and hops of the node.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        infections = s.beta * s.S * s.I / (s.S + s.I + s.R)
        recoveries = s.gamma * s.I
        return [
            Rate(s.S, -infections + s.D * laplacian(s.S)),
            Rate(s.I, infections - recoveries + s.D * laplacian(s.I)),
            Rate(s.R, recoveries + s.D * laplacian(s.R)),
        ]


@register_model(
    field='biology/epidemiology',
    tags=('network', 'mean field', 'ode', 'epidemic threshold', 'scale-free'),
    references=('Pastor-Satorras and Vespignani 2001', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 50.0, 'dt': 0.5},
)
def heterogeneous_mean_field_sis(
    exponent: float = 2.5,
    k_min: int = 3,
    k_max: int = 30,
    *,
    lam: float = 0.2,
    rho: float = 0.01,
) -> System:
    """Return the SIS epidemic on a network with a power-law degree distribution, in heterogeneous mean field.

    The degrees run from `k_min` to `k_max` with `P(k) ~ k^-exponent`. The infection persists above the epidemic
    threshold `lam_c = <k> / <k^2>`, which vanishes on scale-free networks as `k_max` grows, and dies out below it.

    Args:
        exponent:
            The exponent of the degree distribution.

        k_min:
            The smallest degree.

        k_max:
            The largest degree.

        lam:
            The spreading rate, the ratio of the infection and recovery rates.

        rho:
            The initial infected density of every class.

    Returns:
        system:
            The model, with the densities `network.rho`, one per degree from `k_min` to `k_max`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import heterogeneous_mean_field_sis
        >>> len(heterogeneous_mean_field_sis(k_min=2, k_max=5).elements[0].degrees)
        4
    """
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    weights = degrees**-exponent / np.sum(degrees**-exponent)
    classes = DegreeClasses(
        'network',
        rho=[rho] * len(degrees),
        lam=lam,
        degrees=tuple(degrees.tolist()),
        weights=tuple(weights.tolist()),
    )
    return System(elements=[classes], space=EuclideanSpace(len(degrees)))


@register_model(
    field='biology/epidemiology',
    tags=('network', 'metapopulation', 'reaction-diffusion', 'ode', 'field'),
    references=('Colizza, Pastor-Satorras, and Vespignani 2007', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 150.0, 'dt': 1.0},
)
def metapopulation(
    graph: object = None,
    *,
    population: float = 1000.0,
    infected: float = 10.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    D: float = 0.01,
) -> System:
    """Return SIR epidemics in subpopulations at the nodes of a network, linked by individuals who travel.

    The infection starts in the first node and spreads to the others with the travellers. The total number of
    individuals stays constant.

    Args:
        graph:
            The NetworkX graph, or `None` for a Barabási-Albert graph of 30 nodes.

        population:
            The initial number of individuals of every node.

        infected:
            The initial number of infected individuals in the first node.

        beta:
            The infection rate.

        gamma:
            The recovery rate.

        D:
            The rate at which an individual hops along one edge.

    Returns:
        system:
            The model, with the compartments `cities.S`, `cities.I`, and `cities.R`, one value per node.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.biology.epidemiology import metapopulation
        >>> metapopulation(nx.path_graph(4)).elements[0].size
        4
    """
    graph = nx.barabasi_albert_graph(30, 2, seed=0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    sick = np.zeros(size)
    sick[0] = infected
    values = {
        'S': np.full(size, population) - sick,
        'I': sick,
        'R': np.zeros(size),
        'beta': beta,
        'gamma': gamma,
        'D': D,
    }
    return System(elements=[Field('cities', Subpopulation, NetworkSpace(graph), values=values)])
