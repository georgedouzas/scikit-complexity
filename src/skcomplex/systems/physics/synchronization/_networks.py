"""Synchronization on complex networks.

Kuramoto oscillators and coupled logistic maps sit on the edges of a graph.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import Target
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.oscillators import KuramotoCouplingInteraction, LinkElement, PhaseOscillatorElement, order_parameter


class LogisticNodeElement(Element):
    """Logistic map on a node of a network.

    The map has its value, growth rate, and weight of the neighbours.

    `LogisticCouplingInteraction` iterates `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`, where the map
    `f(x) = r x (1 - x)` of the node mixes with the mean of the maps of its neighbours `j` on the network.

    Args:
        label:
            The label of the node or the group on the nodes.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        r:
            The growth rate of the logistic map.

        epsilon:
            The weight of the neighbours.

    Attributes:
        default_visual_:
            The visual of the node when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LogisticNodeElement
        >>> LogisticNodeElement('node').epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        r: Annotated[float, ParameterSpec()] = 4.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
        self.epsilon = epsilon


class EdgeCouplingInteraction(KuramotoCouplingInteraction):
    """Kuramoto coupling along the weighted edges of a network.

    The coupling is `theta_i' += K w_ij sin(theta_j - theta_i)`.

    The coupling runs once per edge of the network of its oscillators and reads the weight `w` of the edge from the
    group on the edges.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators on the nodes.

        links:
            The path of the group on the edges, whose parameter `w` weighs every edge.

        K:
            The coupling strength.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import EdgeCouplingInteraction
        >>> EdgeCouplingInteraction('coupling', target='nodes', links='edges', K=0.5).weights
        'w'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: float = 1.0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(label, target=target, K=K, alpha=alpha, weights='w')
        self.links = links


class LogisticCouplingInteraction(CouplingInteraction):
    """Coupled logistic maps.

    The maps follow `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)` with `f(x) = r x (1 - x)`.

    The map of every node mixes with the mean of the maps of its neighbours `j` on its lattice or network, with the
    parameters `r` and `epsilon` of the node. A node with no neighbours keeps `(1 - epsilon) f(x_i)`. It is the map
    form of coupling, with the mean of the maps of the neighbours combined with the map of the node.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of maps, on the nodes of a lattice or network.

        state:
            The name of the state that the maps iterate.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.elements import LatticeElement
        >>> from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.systems.physics.synchronization import LogisticCouplingInteraction, LogisticNodeElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> maps = LatticeElement(
        ...     'maps',
        ...     element=LogisticNodeElement,
        ...     space=NetworkSpace(nx.path_graph(2)),
        ...     values={'x': [0.25, 0.5], 'epsilon': 0.5},
        ... )
        >>> coupling = LogisticCouplingInteraction('coupling', target='maps')
        >>> system = System(elements=[maps], interactions=[coupling], time=DiscreteTime())
        >>> simulate(system, until=1).values('maps.x')[-1].tolist()
        [0.875, 0.875]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], state: str = 'x') -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._get_map,
            reduce='mean',
            form='map',
            combine=self._get_mix,
        )
        self.state = state

    @staticmethod
    def _logistic(r: Expression, x: Expression) -> Expression:
        """Return the logistic map `r x (1 - x)`."""
        return r * x * (1 - x)

    def _get_map(self, namespace: SimpleNamespace) -> Expression:
        """Return the map of the neighbour, with the growth rate of the node."""
        return self._logistic(namespace.member.r, getattr(namespace.other, self.state))

    def _get_mix(self, namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the map of the node mixed with the mean map of its neighbours."""
        member = namespace.member
        own = self._logistic(member.r, getattr(member, self.state))
        return (1 - member.epsilon) * own + member.epsilon * reduction


NETWORKS = ('watts_strogatz', 'erdos_renyi', 'barabasi_albert')


def _network(name: str, n_nodes: int, degree: int, rewiring: float, seed: int) -> NetworkSpace:
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

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import KuramotoOnNetworksSystem
        >>> system = KuramotoOnNetworksSystem(20, degree=4).prepare().system_
        >>> system.elements[1].prepare().size_
        40
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'oscillators', 'phase', 'network')
    references = ('Arenas et al. 2008', 'Jensen 2023')
    until = 40.0
    dt = 0.2
    observables = tuple(order_parameter('order', 'oscillators'))

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
    ) -> None:
        super().__init__(label)
        self.n_oscillators = n_oscillators
        self.network = network
        self.degree = degree
        self.rewiring = rewiring
        self.K = K
        self.sigma = sigma
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        space = _network(self.network, self.n_oscillators, self.degree, self.rewiring, self.seed)
        rng = np.random.default_rng(self.seed + 1)
        mean_degree = 2 * space.graph.number_of_edges() / self.n_oscillators
        oscillators = LatticeElement(
            'oscillators',
            element=PhaseOscillatorElement,
            space=space,
            values={
                'theta': rng.uniform(0, 2 * math.pi, self.n_oscillators),
                'omega': rng.normal(0.0, self.sigma, self.n_oscillators),
            },
            visual=VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi}),
        )
        links = GroupElement('links', element=LinkElement, space=space, on='edges')
        coupling = EdgeCouplingInteraction('coupling', target='oscillators', links='links', K=self.K / mean_degree)
        return System(self.label, elements=[oscillators, links], interactions=[coupling])


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

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import CoupledMapLatticesOnNetworksSystem
        >>> CoupledMapLatticesOnNetworksSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'map', 'chaos', 'network')
    references = ('Arenas et al. 2008', 'Kaneko 1989')
    until = 100.0
    dt = 1.0
    observables = (ObservableSpec('spread', group='maps', variable='x', reduce='var'),)

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
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.network = network
        self.degree = degree
        self.rewiring = rewiring
        self.r = r
        self.epsilon = epsilon
        self.seed = seed

    def _build_system(self) -> System:
        """Return chaotic logistic maps on the nodes of a network.

        Each map is pulled towards the mean of its neighbours.
        """
        space = _network(self.network, self.n_nodes, self.degree, self.rewiring, self.seed)
        rng = np.random.default_rng(self.seed + 1)
        maps = LatticeElement(
            'maps',
            element=LogisticNodeElement,
            space=space,
            values={'x': rng.uniform(0.0, 1.0, self.n_nodes), 'r': self.r, 'epsilon': self.epsilon},
            visual=VisualSpec('sites', {'states': ['x'], 'low': 0.0, 'high': 1.0}),
        )
        coupling = LogisticCouplingInteraction('coupling', target='maps')
        return System(self.label, elements=[maps], interactions=[coupling], time=DiscreteTime())
