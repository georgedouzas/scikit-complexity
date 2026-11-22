"""Synchronization on complex networks.

Kuramoto oscillators and coupled logistic maps sit on the edges of a graph.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.oscillators import LinkElement, PhaseOscillatorElement, PhaseRotationInteraction, order_parameter

NETWORKS = ('watts_strogatz', 'erdos_renyi', 'barabasi_albert')


class LogisticNodeElement(Element):
    """Logistic maps on the nodes of a network, with their values, growth rates, and weights of the neighbours.

    `LogisticCouplingInteraction` iterates `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`, where the map
    `f(x) = r x (1 - x)` of the node mixes with the mean of the maps of its neighbours `j` on the network.

    Args:
        label:
            The label of the maps.

        x:
            The initial value, or one value per node.

        r:
            The growth rate of the logistic map.

        epsilon:
            The weight of the neighbours.

        space:
            The network or lattice with one map on every node, or `None` for one map.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LogisticNodeElement
        >>> LogisticNodeElement('node').epsilon
        0.5
    """

    visual = VisualSpec('sites', {'states': ['x'], 'low': 0.0, 'high': 1.0})

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.5,
        r: float = 4.0,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.r = r
        self.epsilon = epsilon


class EdgeCouplingInteraction(Interaction):
    """Kuramoto coupling along the weighted edges of a network.

    The coupling is `theta_i' += K w_ij sin(theta_j - theta_i - alpha)` at both ends of every edge.

    The coupling runs once per edge of the network of its oscillators and reads the weight `w` of the edge from the
    element on the edges, as the lines of a power grid or the gap junctions of neurons.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        among:
            The path of the group of oscillators on the nodes.

        along:
            `True` for the edges of the network of the oscillators, or the name of a layer.

        K:
            The coupling strength.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import EdgeCouplingInteraction
        >>> EdgeCouplingInteraction('coupling', among='nodes', K=0.5).K
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        along: bool | str | None = True,
        K: float = 1.0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.K = K
        self.alpha = alpha

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the turn of the phases at both ends of an edge towards each other.

        Args:
            context:
                The namespace of the ends `source` and `target`, the edge `edge`, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        first, second = s.source.theta, s.target.theta
        weight = s.K * s.edge.w
        self.laws_ = [
            RateLaw(second.diff(s.t), weight * sin(first - second - s.alpha)),
            RateLaw(first.diff(s.t), weight * sin(second - first - s.alpha)),
        ]


class LogisticCouplingInteraction(Interaction):
    """Coupled logistic maps.

    The maps follow `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)` with `f(x) = r x (1 - x)`.

    The map of every node mixes with the mean of the maps of its neighbours `j` on its lattice or network, with the
    parameters `r` and `epsilon` of the node. A node with no neighbours keeps `(1 - epsilon) f(x_i)`. It is the map
    form of coupling, as for populations on islands that exchange migrants or chaotic lasers that share light.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the maps, on the nodes of a lattice or network.

        state:
            The name of the state that the maps iterate.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.spaces import NetworkSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.synchronization import LogisticCouplingInteraction, LogisticNodeElement
        >>> maps = LogisticNodeElement('maps', x=[0.25, 0.5], epsilon=0.5, space=NetworkSpace(nx.path_graph(2)))
        >>> system = System(elements=[maps], interactions=[LogisticCouplingInteraction('coupling', target='maps')])
        >>> system.prepare().simulate(until=1).get_values('maps.x')[-1].tolist()
        [0.875, 0.875]
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, state: str = 'x') -> None:
        super().__init__(label)
        self.target = target
        self.state = state

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the map of every node mixed with the mean map of its neighbours.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        node = context.target
        x, other = getattr(node, self.state), getattr(node.other, self.state)
        neighbours = neighbors(node.r * other * (1 - other), reduce='mean')
        self.laws_ = [UpdateLaw(x, (1 - node.epsilon) * (node.r * x * (1 - x)) + node.epsilon * neighbours)]


class NodeOscillatorElement(PhaseOscillatorElement):
    """Phase oscillator on a node, drawn with its phase."""

    visual = VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi})


class KuramotoOnNetworksSystem(BaseSystem):
    """Kuramoto oscillators coupled along the edges of a network.

    The oscillators follow `theta_i' = omega_i + K sum_j A_ij sin(...)`.

    Oscillator `i` follows `theta_i' = omega_i + (K / <k>) sum_j w_ij sin(theta_j - theta_i)` over its neighbours `j`,
    where the weights `w_ij = 1` live on a group on the edges and `<k>` is the mean degree. The coupling runs once per
    edge. The natural frequencies are drawn from a normal distribution of width `sigma`, and the phases uniformly from
    the circle. Above a critical coupling of the order of the all-to-all value `2 sigma sqrt(2 / pi)`, the order
    parameter `r = |<exp(i theta)>|` grows from about `1 / sqrt(N)` towards 1, and small-world shortcuts lower it. The
    system has the phases `oscillators.theta`, the weights `links.w`, and the order parameter `order`.

    Args:
        n_oscillators:
            The number of oscillators, the nodes of the network.

        network:
            The generator, `'watts_strogatz'`, `'erdos_renyi'`, or `'barabasi_albert'`.

        degree:
            The mean degree of the network, even for the Watts-Strogatz ring.

        rewiring:
            The probability of rewiring every edge of the Watts-Strogatz ring.

        K:
            The coupling strength, divided by the mean degree.

        sigma:
            The standard deviation of the natural frequencies.

        seed:
            The seed of the network, the frequencies, and the phases.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import KuramotoOnNetworksSystem
        >>> system = KuramotoOnNetworksSystem(20, degree=4).prepare().system_
        >>> system.elements[1].space.graph.number_of_edges()
        40
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'oscillators', 'phase', 'network')
    references = ('Arenas et al. 2008', 'Jensen 2023')
    until = 40.0
    dt = 0.2
    observables = tuple(order_parameter('order', 'oscillators'))

    @staticmethod
    def _get_network(name: str, n_nodes: int, degree: int, rewiring: float, seed: int) -> NetworkSpace:
        """Return a network of a generator with a mean degree near `degree`."""
        rng = np.random.default_rng(seed)
        if name == 'watts_strogatz':
            return NetworkSpace(nx.watts_strogatz_graph(n_nodes, degree, rewiring, seed=rng))
        if name == 'erdos_renyi':
            return NetworkSpace(nx.gnp_random_graph(n_nodes, degree / (n_nodes - 1), seed=rng))
        if name == 'barabasi_albert':
            return NetworkSpace(nx.barabasi_albert_graph(n_nodes, max(degree // 2, 1), seed=rng))
        error_msg = f'Unknown network {name!r}. Use one of {", ".join(NETWORKS)}.'
        raise ValueError(error_msg)

    def __init__(
        self,
        n_oscillators: int = 200,
        *,
        network: str = 'watts_strogatz',
        degree: int = 6,
        rewiring: float = 0.1,
        K: float = 1.0,
        sigma: float = 0.5,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.network = network
        self.degree = degree
        self.rewiring = rewiring
        self.K = K
        self.sigma = sigma
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Kuramoto oscillators coupled along the edges of a network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        space = KuramotoOnNetworksSystem._get_network(
            self.network,
            self.n_oscillators,
            self.degree,
            self.rewiring,
            self.seed,
        )
        rng = np.random.default_rng(self.seed + 1)
        mean_degree = 2 * space.graph.number_of_edges() / self.n_oscillators
        oscillators = NodeOscillatorElement(
            'oscillators',
            theta=rng.uniform(0, 2 * math.pi, self.n_oscillators),
            omega=rng.normal(0.0, self.sigma, self.n_oscillators),
            space=space,
        )
        links = LinkElement('links', space=space, on='edges')
        interactions = [
            PhaseRotationInteraction('rotation', target='oscillators'),
            EdgeCouplingInteraction('coupling', among='oscillators', K=self.K / mean_degree),
        ]
        self.system_ = System(self.label, elements=[oscillators, links], interactions=interactions)


class CoupledMapLatticesOnNetworksSystem(BaseSystem):
    """Chaotic logistic maps on the nodes of a network.

    Each map is pulled towards the mean of its neighbours.

    Node `i` follows `x_i(t + 1) = (1 - epsilon) f(x_i) + (epsilon / k_i) sum_j A_ij f(x_j)` with the logistic map `f(x)
    = r x (1 - x)`, a mean over the neighbours of the network. For a coupling `epsilon` large enough against the
    Lyapunov exponent of the map and the spectral gap of the network, the maps synchronize on one chaotic orbit, and the
    variance of the states over the nodes falls to zero. Below it they stay incoherent. The system has the states
    `maps.x` and their variance over the nodes, `spread`.

    Args:
        n_nodes:
            The number of maps, the nodes of the network.

        network:
            The generator, `'watts_strogatz'`, `'erdos_renyi'`, or `'barabasi_albert'`.

        degree:
            The mean degree of the network.

        rewiring:
            The probability of rewiring every edge of the Watts-Strogatz ring.

        r:
            The parameter of the logistic map, chaotic at 4.

        epsilon:
            The coupling between 0 and 1.

        seed:
            The seed of the network and the initial states.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import CoupledMapLatticesOnNetworksSystem
        >>> CoupledMapLatticesOnNetworksSystem(20).prepare().system_.elements[0].space.graph.number_of_nodes()
        20
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'map', 'chaos', 'network')
    references = ('Arenas et al. 2008', 'Kaneko 1989')
    until = 100.0
    dt = 1.0
    observables = (Observable('spread', group='maps', variable='x', reduce='var'),)

    def __init__(
        self,
        n_nodes: int = 100,
        *,
        network: str = 'erdos_renyi',
        degree: int = 20,
        rewiring: float = 0.1,
        r: float = 4.0,
        epsilon: float = 0.9,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.network = network
        self.degree = degree
        self.rewiring = rewiring
        self.r = r
        self.epsilon = epsilon
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of chaotic logistic maps on the nodes of a network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValueError:
                The network is unknown.
        """
        super()._prepare(context)
        space = KuramotoOnNetworksSystem._get_network(self.network, self.n_nodes, self.degree, self.rewiring, self.seed)
        rng = np.random.default_rng(self.seed + 1)
        maps = LogisticNodeElement(
            'maps',
            x=rng.uniform(0.0, 1.0, self.n_nodes),
            r=self.r,
            epsilon=self.epsilon,
            space=space,
        )
        coupling = LogisticCouplingInteraction('coupling', target='maps')
        self.system_ = System(self.label, elements=[maps], interactions=[coupling])
