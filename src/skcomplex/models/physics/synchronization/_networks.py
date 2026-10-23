"""Synchronization on complex networks: Kuramoto oscillators and coupled logistic maps on the edges of a graph."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, Expression, VisualSpec, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Law, Symbols, Update, neighbors
from ....core.laws import Rate as RateLaw
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import Observable, System
from ._kuramoto import PhaseOscillator
from ._utils import _order_parameter

NETWORKS = ('watts_strogatz', 'erdos_renyi', 'barabasi_albert')

Link = Element.define('Link', parameters={'w': parameter(default=1.0, domain=Domain.NONNEGATIVE)})


def _logistic(r: Expression, x: Expression) -> Expression:
    """Return the logistic map `r x (1 - x)`."""
    return r * x * (1 - x)


LogisticNode = Element.define(
    'LogisticNode',
    states={'x': state(default=0.5, domain=Domain.REAL)},
    parameters={'r': 4.0, 'epsilon': parameter(default=0.5, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Update(
            s.x,
            (1 - s.epsilon) * _logistic(s.r, s.x) + s.epsilon * neighbors(_logistic(s.r, s.other.x), reduce='mean'),
        ),
    ],
)


class EdgeCoupling(Interaction):
    """Kuramoto coupling along one weighted edge, `theta_i' += K w_ij sin(theta_j - theta_i)` at both of its ends.

    The interaction runs once per edge of the network of its oscillators and reads the weight `w` of the edge from
    the group on the edges.

    Args:
        label:
            The label of the interaction.

        oscillators:
            The path of the group of oscillators on the nodes.

        links:
            The path of the group on the edges, whose parameter `w` weighs every edge.

        K:
            The coupling strength.

    Examples:
        >>> from skcomplex.models.physics.synchronization import EdgeCoupling
        >>> EdgeCoupling('coupling', oscillators='nodes', links='edges', K=0.5).K
        0.5
    """

    over: ClassVar[str] = 'neighbours'
    oscillators: str = role()
    links: str = role()

    K: float = parameter(default=1.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the pull of both ends of an edge towards each other.

        Args:
            s:
                The namespace of the edge, with its ends `s.source` and `s.target` and its member `s.edge`.

        Returns:
            laws:
                The rates of the phases at both ends.
        """
        pull = s.K * s.edge.w * sin(s.target.theta - s.source.theta)
        return [RateLaw(s.source.theta, pull), RateLaw(s.target.theta, -pull)]


def _network(name: str, n_nodes: int, degree: int, rewiring: float, seed: int) -> NetworkSpace:
    """Return a network of a generator with a mean degree near `degree`."""
    rng = np.random.default_rng(seed)
    if name == 'watts_strogatz':
        return NetworkSpace.watts_strogatz(n_nodes, degree, rewiring, seed=rng)
    if name == 'erdos_renyi':
        return NetworkSpace.erdos_renyi(n_nodes, degree / (n_nodes - 1), seed=rng)
    if name == 'barabasi_albert':
        return NetworkSpace.barabasi_albert(n_nodes, max(degree // 2, 1), seed=rng)
    error_msg = f'Unknown network {name!r}. Use one of {", ".join(NETWORKS)}.'
    raise ValueError(error_msg)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'network'),
    references=('Arenas et al. 2008', 'Jensen 2023'),
    defaults={'until': 40.0, 'dt': 0.2},
    observables=lambda system: _order_parameter('order', 'oscillators'),
)
def kuramoto_on_networks(
    n_oscillators: int = 200,
    *,
    network: str = 'watts_strogatz',
    degree: int = 6,
    rewiring: float = 0.1,
    K: float = 1.0,
    sigma: float = 0.5,
    seed: int = 0,
) -> System:
    """Return Kuramoto oscillators coupled along the edges of a network, `theta_i' = omega_i + K sum_j A_ij sin(...)`.

    Oscillator `i` follows `theta_i' = omega_i + (K / <k>) sum_j w_ij sin(theta_j - theta_i)` over its neighbours `j`,
    where the weights `w_ij = 1` live on a group on the edges and `<k>` is the mean degree. The coupling runs once per
    edge. The natural frequencies are drawn from a normal distribution of width `sigma`, and the phases uniformly from
    the circle. Above a critical coupling of the order of the all-to-all value `2 sigma sqrt(2 / pi)`, the order
    parameter `r = |<exp(i theta)>|` grows from about `1 / sqrt(N)` towards 1, and small-world shortcuts lower it.

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

    Returns:
        system:
            The model, with the phases `oscillators.theta`, the weights `links.w`, and the order parameter `order`.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.models.physics.synchronization import kuramoto_on_networks
        >>> system = kuramoto_on_networks(20, degree=4)
        >>> system.components[1].size
        40
    """
    space = _network(network, n_oscillators, degree, rewiring, seed)
    rng = np.random.default_rng(seed + 1)
    mean_degree = 2 * space.graph.number_of_edges() / n_oscillators
    oscillators = PhaseOscillator.lattice(
        'oscillators',
        space,
        theta=rng.uniform(0, 2 * math.pi, n_oscillators),
        omega=rng.normal(0.0, sigma, n_oscillators),
        visual=VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi}),
    )
    links = Group('links', element=Link, space=space, on='edges')
    coupling = EdgeCoupling('coupling', oscillators='oscillators', links='links', K=K / mean_degree)
    return System(components=[oscillators, links], interactions=[coupling])


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'map', 'chaos', 'network'),
    references=('Arenas et al. 2008', 'Kaneko 1989'),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: [Observable('spread', group='maps', variable='x', reduce='var')],
)
def coupled_map_lattices_on_networks(
    n_nodes: int = 100,
    *,
    network: str = 'erdos_renyi',
    degree: int = 20,
    rewiring: float = 0.1,
    r: float = 4.0,
    epsilon: float = 0.9,
    seed: int = 0,
) -> System:
    """Return chaotic logistic maps on the nodes of a network, each pulled towards the mean of its neighbours.

    Node `i` follows `x_i(t + 1) = (1 - epsilon) f(x_i) + (epsilon / k_i) sum_j A_ij f(x_j)` with the logistic map
    `f(x) = r x (1 - x)`, a mean over the neighbours of the network. For a coupling `epsilon` large enough against the
    Lyapunov exponent of the map and the spectral gap of the network, the maps synchronize on one chaotic orbit, and
    the variance of the states over the nodes falls to zero. Below it they stay incoherent.

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

    Returns:
        system:
            The model, with the states `maps.x` and their variance over the nodes, `spread`.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.models.physics.synchronization import coupled_map_lattices_on_networks
        >>> coupled_map_lattices_on_networks(20).components[0].size
        20
    """
    space = _network(network, n_nodes, degree, rewiring, seed)
    rng = np.random.default_rng(seed + 1)
    maps = LogisticNode.lattice(
        'maps',
        space,
        x=rng.uniform(0.0, 1.0, n_nodes),
        r=r,
        epsilon=epsilon,
        visual=VisualSpec('sites', {'states': ['x'], 'low': 0.0, 'high': 1.0}),
    )
    return System(components=[maps], time=DiscreteTime())
