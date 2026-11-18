"""Adaptive networks of phase oscillators.

Their coupling weights evolve with the phases they couple.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import sin

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ..common.oscillators import PhaseOscillatorElement, PhaseRotationInteraction


class AdaptiveLinkElement(Element):
    """Directed link of an adaptive network with the coupling weight `kappa`.

    The weight is a state that the phases of its ends adapt.

    Args:
        label:
            The label of the link or the group on the links.

        kappa:
            The initial coupling weight, or one weight per link.

        space:
            The network of the links, or `None` for one link.

        on:
            `'edges'` for one link on every edge of the network, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveLinkElement
        >>> AdaptiveLinkElement('link', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        kappa: object = 0.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.kappa = kappa


class AdaptiveCouplingInteraction(Interaction):
    """Adaptive Kuramoto-Sakaguchi coupling.

    The weights relax towards a phase-lagged sine of the phase difference.

    On every directed link from `j` to `i`, the phase of `i` turns by `-K kappa_ij sin(theta_i - theta_j + alpha)` and
    the weight relaxes as `kappa_ij' = -epsilon (kappa_ij + sin(theta_i - theta_j + beta))`, so it grows between
    oscillators whose phases differ by about `-beta - pi / 2`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        among:
            The path of the group of phase oscillators on the nodes, whose edges hold the adaptive links.

        along:
            `True` for the edges of the network of the oscillators, or the name of a layer.

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
        >>> AdaptiveCouplingInteraction('adaptation', among='nodes').epsilon
        0.01
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
        beta: float = 0.0,
        epsilon: float = 0.01,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the coupling of every link's target and the adaptation of its weight.

        Args:
            context:
                The namespace of the ends `source` and `target`, the link `edge`, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            difference = s.target.theta - s.source.theta
            self.laws_ = [
                RateLaw(s.target.theta.diff(s.t), -s.K * s.edge.kappa * sin(difference + s.alpha)),
                RateLaw(s.edge.kappa.diff(s.t), -s.epsilon * (s.edge.kappa + sin(difference + s.beta))),
            ]


class NodeOscillatorElement(PhaseOscillatorElement):
    """Phase oscillator on a node, drawn with its phase."""

    visual = VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi})


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AdaptiveKuramotoSystem
        >>> AdaptiveKuramotoSystem(5).prepare().system_.elements[1].space.graph.number_of_edges()
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.omega = omega
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of globally coupled phase oscillators whose weights adapt to the phase differences.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        graph = nx.complete_graph(self.n_oscillators, create_using=nx.DiGraph)
        space = NetworkSpace(graph)
        nodes = NodeOscillatorElement(
            'nodes',
            theta=rng.uniform(0.0, 2 * math.pi, self.n_oscillators),
            omega=self.omega,
            space=space,
        )
        links = AdaptiveLinkElement(
            'links',
            kappa=rng.uniform(-1.0, 1.0, graph.number_of_edges()),
            space=space,
            on='edges',
        )
        interactions = [
            PhaseRotationInteraction('rotation', target='nodes'),
            AdaptiveCouplingInteraction(
                'adaptation',
                among='nodes',
                K=1 / self.n_oscillators,
                alpha=self.alpha,
                beta=self.beta,
                epsilon=self.epsilon,
            ),
        ]
        self.system_ = System(self.label, elements=[nodes, links], interactions=interactions)
