"""Adaptive networks of phase oscillators, whose coupling weights evolve with the phases they couple."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import sin

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.oscillators import PhaseOscillatorElement


class AdaptiveLinkElement(Element):
    """Directed link of an adaptive network with the coupling weight `kappa`, a state that the phases of its ends adapt.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        kappa:
            The initial coupling weight.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveLinkElement
        >>> AdaptiveLinkElement('link', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa


class AdaptiveCouplingInteraction(NeighborInteraction):
    """Adaptive Kuramoto-Sakaguchi coupling, whose weights relax towards a phase-lagged sine of the phase difference.

    On every directed link from `j` to `i`, the phase of `i` turns by `-K kappa_ij sin(theta_i - theta_j + alpha)` and
    the weight relaxes as `kappa_ij' = -epsilon (kappa_ij + sin(theta_i - theta_j + beta))`, so it grows between
    oscillators whose phases differ by about `-beta - pi / 2`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group of phase oscillators on the nodes.

        links:
            The path of the group of adaptive links on the edges.

        K:
            The coupling strength, `1 / N` for the mean coupling of `N` oscillators.

        alpha:
            The phase lag of the coupling.

        beta:
            The phase lag of the adaptation.

        epsilon:
            The rate of the adaptation.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveCouplingInteraction
        >>> AdaptiveCouplingInteraction('adaptation', nodes='nodes', links='links').epsilon
        0.01
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.01,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the coupling of the target of every link and the adaptation of its weight.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the ends `source` and `target` and the `edge`.

        Returns:
            laws:
                The rate law of the phase of the target and the rate law of the weight of the link.
        """
        s = namespace
        difference = s.target.theta - s.source.theta
        return [
            RateLaw(s.target.theta, -s.K * s.edge.kappa * sin(difference + s.alpha)),
            RateLaw(s.edge.kappa, -s.epsilon * (s.edge.kappa + sin(difference + s.beta))),
        ]


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
