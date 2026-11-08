"""Adaptive networks of phase oscillators.

Their coupling weights evolve with the phases they couple.
"""

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
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.oscillators import PhaseOscillatorElement


class AdaptiveLinkElement(Element):
    """Directed link of an adaptive network with the coupling weight `kappa`.

    The weight is a state that the phases of its ends adapt.

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
    """Adaptive Kuramoto-Sakaguchi coupling.

    The weights relax towards a phase-lagged sine of the phase difference.

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

    law_classes = (RateLaw,)

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
        """Return the coupling of every link's target and the adaptation of its weight.

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


class AdaptiveKuramotoSystem(BaseSystem):
    """Globally coupled phase oscillators with adaptive coupling weights.

    The weights adapt to the phase differences.

    The oscillators follow the Kuramoto-Sakaguchi model `theta_i' = omega - (1 / N) Σ_j kappa_ij sin(theta_i - theta_j +
    alpha)`, and every weight of the directed link from `j` to `i` adapts as `kappa_ij' = -epsilon (kappa_ij +
    sin(theta_i - theta_j + beta))`, Eqs. 28-29 of Berner et al. The weights are states of a group on the edges of the
    complete directed graph. Even identical oscillators split into frequency clusters: groups whose members share one
    mean frequency, strongly coupled within and weakly coupled across, with the weights inside a cluster near
    `-sin(theta_i - theta_j + beta)`. The phases and the weights start uniformly at random in `[0, 2 pi)` and `[-1, 1]`.
    The system has the phases `nodes.theta` and the weights `links.kappa`, one per directed link.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveKuramotoSystem
        >>> AdaptiveKuramotoSystem(5).prepare().system_.elements[1].prepare().size_
        20
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'adaptive network', 'frequency clusters', 'phase', 'ode')
    references = ('Berner et al. 2023', 'Berner et al. 2019', 'Kasatkin et al. 2017')
    until = 2000.0
    dt = 1.0

    def __init__(
        self,
        n_oscillators: int = 50,
        *,
        alpha: float = 0.3 * math.pi,
        beta: float = 0.23 * math.pi,
        epsilon: float = 0.01,
        omega: float = 0.0,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_oscillators = n_oscillators
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.omega = omega
        self.seed = seed

    def _build_system(self) -> System:
        """Return globally coupled phase oscillators with adaptive coupling weights.

        The weights adapt to the phase differences.
        """
        rng = np.random.default_rng(self.seed)
        graph = nx.complete_graph(self.n_oscillators, create_using=nx.DiGraph)
        space = NetworkSpace(graph)
        nodes = GroupElement(
            'nodes',
            element=PhaseOscillatorElement,
            space=space,
            on='sites',
            values={'theta': rng.uniform(0.0, 2 * math.pi, self.n_oscillators), 'omega': self.omega},
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
            K=1 / self.n_oscillators,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.epsilon,
        )
        return System(self.label, elements=[nodes, links], interactions=[adaptation])
