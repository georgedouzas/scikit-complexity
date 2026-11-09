"""Epidemiology systems: infection on networks.

The infection is in the mean-field limit and between subpopulations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import CouplingInteraction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DegreeClassElement(Element):
    """Infected density of the nodes of one degree, in heterogeneous mean field.

    A node of degree `k` recovers at rate 1 and is infected at rate `lam k Theta`, where
    `Theta = sum_k k P(k) rho_k / <k>` is the probability that an edge points to an infected node, a total over the
    degree classes, so `d rho_k / dt = -rho_k + lam k (1 - rho_k) Theta`. The classes are the members of a group, and
    the recovery and the infection are interactions of the group, which `HeterogeneousMeanFieldSISSystem` adds.

    Args:
        label:
            The label of the class or the group of classes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rho:
            The initial infected density.

        lam:
            The spreading rate, the ratio of the infection and recovery rates.

        k:
            The degree of the class.

        P:
            The probability `P(k)` of the degree.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import DegreeClassElement
        >>> GroupElement(
        ...     'network', element=DegreeClassElement, size=2, values={'k': [1.0, 2.0], 'P': [0.5, 0.5]}
        ... ).prepare().size_
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        P: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.lam = lam
        self.k = k
        self.P = P


class SubpopulationElement(Element):
    """Well-mixed SIR population at a node of a network.

    Its individuals hop along each edge at the rate `D`.

    Inside the node, `beta S I / N` individuals are infected and `gamma I` recover per unit time, with
    `N = S + I + R`. Hopping along every edge adds `D` times the graph Laplacian to every compartment.

    Args:
        label:
            The label of the subpopulation or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.epidemiology import SubpopulationElement
        >>> SubpopulationElement('city', S=1000.0).beta
        0.3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        I: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        R: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.I = I
        self.R = R
        self.beta = beta
        self.gamma = gamma
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.S, -infections + s.D * laplacian(s.S)),
            RateLaw(s.I, infections - recoveries + s.D * laplacian(s.I)),
            RateLaw(s.R, recoveries + s.D * laplacian(s.R)),
        ]


class HeterogeneousMeanFieldSISSystem(BaseSystem):
    """The SIS epidemic on a power-law network, in heterogeneous mean field.

    The degrees run from `k_min` to `k_max` with `P(k) ~ k^-exponent`. The infection persists above the epidemic
    threshold `lam_c = <k> / <k^2>`, which vanishes on scale-free networks as `k_max` grows, and dies out below it. The
    system has the densities `network.rho` of the degree classes from `k_min` to `k_max`, members of the group
    `network`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import HeterogeneousMeanFieldSISSystem
        >>> HeterogeneousMeanFieldSISSystem(k_min=2, k_max=5).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'biology/epidemiology'
    topics = ('network', 'mean field', 'ode', 'epidemic threshold', 'scale-free')
    references = ('Pastor-Satorras and Vespignani 2001', 'Pastor-Satorras et al. 2015')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        exponent: float = 2.5,
        k_min: int = 3,
        k_max: int = 30,
        *,
        lam: float = 0.2,
        rho: float = 0.01,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.exponent = exponent
        self.k_min = k_min
        self.k_max = k_max
        self.lam = lam
        self.rho = rho

    @staticmethod
    def _recovery(namespace: SimpleNamespace) -> Expression:
        """Return the recovery `-rho_k` of a degree class at rate 1."""
        return -namespace.target.rho

    @staticmethod
    def _edge_weight(namespace: SimpleNamespace) -> Expression:
        """Return the share `k P(k)` of the ends of edges that another degree class holds."""
        return namespace.other.k * namespace.other.P

    @staticmethod
    def _infection(namespace: SimpleNamespace) -> Expression:
        """Return the infection `lam k (1 - rho_k) rho_k'` of a degree class.

        The infection is by the nodes of another class.
        """
        member = namespace.member
        return member.lam * member.k * (1 - member.rho) * namespace.other.rho

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the SIS epidemic on a network with power-law degrees."""
        degrees = np.arange(self.k_min, self.k_max + 1, dtype=float)
        weights = degrees**-self.exponent / np.sum(degrees**-self.exponent)
        classes = GroupElement(
            'network',
            element=DegreeClassElement,
            size=len(degrees),
            values={'rho': self.rho, 'lam': self.lam, 'k': degrees, 'P': weights},
        )
        interactions = [
            RateInteraction('recovery', target='network', state='rho', rate=self._recovery),
            CouplingInteraction(
                'infection',
                target='network',
                read='rho',
                write='rho',
                function=self._infection,
                reduce='normalized-mean',
                mode='group',
                weights=self._edge_weight,
                include_self=True,
            ),
        ]
        return CompositeSystem(self.label, elements=[classes], interactions=interactions)


class MetapopulationSystem(BaseSystem):
    """SIR epidemics in subpopulations at the nodes of a network.

    Individuals who travel link the subpopulations.

    The infection starts in the first node and spreads to the others with the travellers. The total number of
    individuals stays constant. The system has the compartments `cities.S`, `cities.I`, and `cities.R`, one value per
    node.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.epidemiology import MetapopulationSystem
        >>> MetapopulationSystem(nx.path_graph(4)).prepare().system_.elements[0].prepare().size_
        4
    """

    field = 'biology/epidemiology'
    topics = ('network', 'metapopulation', 'reaction-diffusion', 'ode', 'field')
    references = ('Colizza, Pastor-Satorras, and Vespignani 2007', 'Pastor-Satorras et al. 2015')
    until = 150.0
    dt = 1.0

    def __init__(
        self,
        graph: object = None,
        *,
        population: float = 1000.0,
        infected: float = 10.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        D: float = 0.01,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.population = population
        self.infected = infected
        self.beta = beta
        self.gamma = gamma
        self.D = D

    def _build_system(self) -> CompositeSystem:
        """Return the parts of SIR epidemics in subpopulations at the nodes of a network."""
        graph = self.graph
        graph = nx.barabasi_albert_graph(30, 2, seed=0) if graph is None else graph
        size = nx.number_of_nodes(graph)
        sick = np.zeros(size)
        sick[0] = self.infected
        values = {
            'S': np.full(size, self.population) - sick,
            'I': sick,
            'R': np.zeros(size),
            'beta': self.beta,
            'gamma': self.gamma,
            'D': self.D,
        }
        return CompositeSystem(
            self.label,
            elements=[FieldElement('cities', element=SubpopulationElement, space=NetworkSpace(graph), values=values)],
        )
