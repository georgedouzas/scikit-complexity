"""Epidemiology systems: infection on networks.

The infection is in the mean-field limit and between subpopulations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian, total
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ...general.dynamics import NetworkDynamicalSystem
from ...general.dynamics.types import Equations


class DegreeClassElement(Element):
    """Infected density of the nodes of one degree, in heterogeneous mean field.

    A node of degree `k` recovers at rate 1 and is infected at rate `lam k Theta`, where
    `Theta = sum_k k P(k) rho_k / <k>` is the probability that an edge points to an infected node, a total over the
    degree classes, so `d rho_k / dt = -rho_k + lam k (1 - rho_k) Theta`. The classes are the members of a group, and
    the recovery and the infection are interactions of the group, which `HeterogeneousMeanFieldSISSystem` adds.

    Args:
        label:
            The label of the class or the group of classes.

        rho:
            The initial infected density.

        lam:
            The spreading rate, the ratio of the infection and recovery rates.

        k:
            The degree of the class.

        P:
            The probability `P(k)` of the degree.

        size:
            The number of degree classes, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import DegreeClassElement
        >>> DegreeClassElement('network', k=[1.0, 2.0], P=[0.5, 0.5], size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        *,
        rho: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.01,
        lam: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.2,
        k: Annotated[float | Array, RealSet.open(0, oo)] = 1.0,
        P: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.rho = rho
        self.lam = lam
        self.k = k
        self.P = P


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import HeterogeneousMeanFieldSISSystem
        >>> HeterogeneousMeanFieldSISSystem(k_min=2, k_max=5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.exponent = exponent
        self.k_min = k_min
        self.k_max = k_max
        self.lam = lam
        self.rho = rho

    @staticmethod
    def _get_recovery(namespace: SimpleNamespace) -> Expression:
        """Return the recovery `-rho_k` of a degree class at rate 1."""
        return -namespace.target.rho

    @staticmethod
    def _get_edge_weight(namespace: SimpleNamespace) -> Expression:
        """Return the share `k P(k)` of the ends of edges that another degree class holds."""
        return namespace.other.k * namespace.other.P

    @staticmethod
    def _get_infection(namespace: SimpleNamespace) -> Expression:
        """Return the infection `lam k (1 - rho_k) rho_k'` of a degree class.

        The infection is by the nodes of another class.
        """
        member = namespace.member
        return member.lam * member.k * (1 - member.rho) * namespace.other.rho

    @staticmethod
    def _get_infection_laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection of every degree class by every class, weighted by the share of the ends of edges."""
        pair = SimpleNamespace(member=s.target, other=s.target.other)
        infection = HeterogeneousMeanFieldSISSystem._get_infection(pair)
        weight = HeterogeneousMeanFieldSISSystem._get_edge_weight(pair)
        mean = total(weight * infection, over=s.target.other) / total(weight, over=s.target.other)
        return [RateLaw(s.target.rho, mean)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the SIS epidemic on a network with power-law degrees.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        degrees = np.arange(self.k_min, self.k_max + 1, dtype=float)
        weights = degrees**-self.exponent / np.sum(degrees**-self.exponent)
        classes = DegreeClassElement('network', rho=self.rho, lam=self.lam, k=degrees, P=weights, size=len(degrees))
        interactions = [
            Interaction(
                'recovery',
                roles={'target': 'network'},
                laws=lambda s: [RateLaw(s.target.rho, self._get_recovery(s))],
            ),
            Interaction('infection', roles={'target': 'network'}, laws=self._get_infection_laws),
        ]
        self.system_ = System(self.label, elements=[classes], interactions=interactions)


class MetapopulationSystem(NetworkDynamicalSystem):
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
        >>> MetapopulationSystem(nx.path_graph(4)).prepare().simulate(until=0.0).get_values('cities.S').shape
        (1, 4)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.graph = graph
        self.population = population
        self.infected = infected
        self.beta = beta
        self.gamma = gamma
        self.D = D

    def _get_graph(self) -> nx.Graph:
        """Return the graph, a Barabási-Albert graph of 30 nodes by default."""
        return nx.barabasi_albert_graph(30, 2, seed=0) if self.graph is None else self.graph

    def _get_element_label(self) -> str:
        """Return the label of the field of the nodes."""
        return 'cities'

    def _get_states(self) -> dict[str, object]:
        """Return the compartments, the infected ones in the first node."""
        size = nx.number_of_nodes(self._get_graph())
        sick = np.zeros(size)
        sick[0] = self.infected
        return {'S': np.full(size, self.population) - sick, 'I': sick, 'R': np.zeros(size)}

    def _get_parameters(self) -> dict[str, object]:
        """Return the rates."""
        return {'beta': self.beta, 'gamma': self.gamma, 'D': self.D}

    def _get_equations(self) -> Equations:
        """Return the infections, recoveries, and hops of every node."""
        return {
            'S': lambda s: -self._get_infections(s.target) + s.target.D * laplacian(s.target.S),
            'I': lambda s: (
                self._get_infections(s.target) - s.target.gamma * s.target.I + s.target.D * laplacian(s.target.I)
            ),
            'R': lambda s: s.target.gamma * s.target.I + s.target.D * laplacian(s.target.R),
        }

    @staticmethod
    def _get_infections(cities: SimpleNamespace) -> Expression:
        """Return the new infections per unit time of a node."""
        return cities.beta * cities.S * cities.I / (cities.S + cities.I + cities.R)
