"""Adaptive networks of phase oscillators, whose coupling weights evolve with the phases they couple."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ..common.oscillators import PhaseOscillatorElement
from ._adaptive_coupling import AdaptiveCouplingInteraction
from ._adaptive_link import AdaptiveLinkElement


@register_system(
    field='physics/synchronization',
    tags=('synchronization', 'adaptive network', 'frequency clusters', 'phase', 'ode'),
    references=('Berner et al. 2023', 'Berner et al. 2019', 'Kasatkin et al. 2017'),
    defaults={'until': 2000.0, 'dt': 1.0},
)
def adaptive_kuramoto(
    n_oscillators: int = 50,
    *,
    alpha: float = 0.3 * math.pi,
    beta: float = 0.23 * math.pi,
    epsilon: float = 0.01,
    omega: float = 0.0,
    seed: int = 0,
) -> System:
    """Return globally coupled phase oscillators whose coupling weights adapt to their phase differences.

    The oscillators follow the Kuramoto-Sakaguchi model `theta_i' = omega - (1 / N) Σ_j kappa_ij sin(theta_i - theta_j +
    alpha)`, and every weight of the directed link from `j` to `i` adapts as
    `kappa_ij' = -epsilon (kappa_ij + sin(theta_i - theta_j + beta))`, Eqs. 28-29 of Berner et al. The weights are
    states of a group on the edges of the complete directed graph. Even identical oscillators split into frequency
    clusters: groups whose members share one mean frequency, strongly coupled within and weakly coupled across, with
    the weights inside a cluster near `-sin(theta_i - theta_j + beta)`. The phases and the weights start uniformly at
    random in `[0, 2 pi)` and `[-1, 1]`.

    Args:
        n_oscillators:
            The number of oscillators.

        alpha:
            The phase lag of the coupling.

        beta:
            The phase lag of the adaptation.

        epsilon:
            The rate of the adaptation, slow for small values.

        omega:
            The natural frequency of every oscillator.

        seed:
            The seed of the random initial phases and weights.

    Returns:
        system:
            The system, with the phases `nodes.theta` and the weights `links.kappa`, one per directed link.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import adaptive_kuramoto
        >>> adaptive_kuramoto(5).elements[1].prepare().size_
        20
    """
    rng = np.random.default_rng(seed)
    graph = nx.complete_graph(n_oscillators, create_using=nx.DiGraph)
    space = NetworkSpace(graph)
    nodes = GroupElement(
        'nodes',
        element=PhaseOscillatorElement,
        space=space,
        on='sites',
        values={'theta': rng.uniform(0.0, 2 * math.pi, n_oscillators), 'omega': omega},
        visual=VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi}),
    )
    links = GroupElement(
        'links',
        element=AdaptiveLinkElement,
        space=space,
        on='edges',
        values={'kappa': rng.uniform(-1.0, 1.0, graph.number_of_edges())},
    )
    adaptation = AdaptiveCouplingInteraction(
        'adaptation',
        nodes='nodes',
        links='links',
        K=1 / n_oscillators,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
    )
    return System(elements=[nodes, links], interactions=[adaptation])
