"""Epidemiology models: infection on networks in the mean-field limit and between subpopulations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import FieldElement, GroupElement
from ....core.interactions import CouplingInteraction, RateInteraction
from ....core.spaces import NetworkSpace
from ....core.system import System
from ..._catalog import register_model
from ._degree_class import DegreeClassElement
from ._subpopulation import SubpopulationElement


def _recovery(namespace: SimpleNamespace) -> Expression:
    """Return the recovery `-rho_k` of a degree class at rate 1."""
    return -namespace.target.rho


def _edge_weight(namespace: SimpleNamespace) -> Expression:
    """Return the share `k P(k)` of the ends of edges that another degree class holds."""
    return namespace.other.k * namespace.other.P


def _infection(namespace: SimpleNamespace) -> Expression:
    """Return the infection `lam k (1 - rho_k) rho_k'` of a degree class by the nodes of another class."""
    member = namespace.member
    return member.lam * member.k * (1 - member.rho) * namespace.other.rho


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
            The model, with the densities `network.rho` of the degree classes from `k_min` to `k_max`, members of the
            group `network`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import heterogeneous_mean_field_sis
        >>> heterogeneous_mean_field_sis(k_min=2, k_max=5).elements[0].check().prepare().size_
        4
    """
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    weights = degrees**-exponent / np.sum(degrees**-exponent)
    classes = GroupElement(
        'network',
        element=DegreeClassElement,
        size=len(degrees),
        values={'rho': rho, 'lam': lam, 'k': degrees, 'P': weights},
    )
    interactions = [
        RateInteraction('recovery', target='network', state='rho', rate=_recovery),
        CouplingInteraction(
            'infection',
            target='network',
            read='rho',
            write='rho',
            function=_infection,
            reduce='normalized-mean',
            mode='group',
            weights=_edge_weight,
            include_self=True,
        ),
    ]
    return System(elements=[classes], interactions=interactions)


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
        >>> metapopulation(nx.path_graph(4)).elements[0].check().prepare().size_
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
    return System(
        elements=[FieldElement('cities', element=SubpopulationElement, space=NetworkSpace(graph), values=values)],
    )
