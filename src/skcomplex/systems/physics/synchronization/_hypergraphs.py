"""Synchronization on hypergraphs.

Kuramoto oscillators are coupled in pairs and through the triangles of a hypergraph.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import sin
from scipy import stats

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import hyperedges
from ....core.components.spaces import NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.oscillators import PhaseRotationInteraction, order_parameter

PAIR = 2
TRIANGLE = 3


class SimplexCouplingInteraction(Interaction):
    """Kuramoto coupling through the simplices of a hypergraph, pairs and triangles.

    On a link, the focal oscillator `i` gets `K1 sin(theta_j - theta_i)`, and on a triangle `K2 sin(theta_j + theta_k -
    2 theta_i)` from its two other members `j` and `k`, as for neurons that fire in groups or species that compete in
    threes.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of oscillators on the nodes of a network with hyperedges.

        K1:
            The coupling of every link.

        K2:
            The coupling of every triangle.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SimplexCouplingInteraction
        >>> SimplexCouplingInteraction('coupling', target='oscillators', K2=0.5).K2
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, K1: float = 0.0, K2: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.K1 = K1
        self.K2 = K2

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the coupling of every oscillator to the rest of its links and triangles.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        theta, others = context.target.theta, context.target.others
        links = hyperedges(context.K1 * sin(others[0].theta - theta), size=PAIR)
        triangles = hyperedges(context.K2 * sin(others[0].theta + others[1].theta - 2 * theta), size=TRIANGLE)
        self.laws_ = [RateLaw(theta.diff(context.t), links + triangles)]


class HypergraphKuramotoSystem(BaseSystem):
    """Kuramoto oscillators coupled in pairs and by three-body interactions.

    The three-body interactions act on the triangles of a hypergraph.

    Oscillator `i` follows `theta_i' = omega_i + (K1 / <k1>) sum_j sin(theta_j - theta_i) + (K2 / <k2>) sum_jk
    sin(theta_j + theta_k - 2 theta_i)` over the links `ij` and the triangles `ijk` of a random hypergraph, where `<k1>`
    and `<k2>` are the mean numbers of links and of triangles of a node. All-to-all, the triangle term is the three-body
    coupling of Skardal and Arenas, and the Ott-Antonsen reduction of Lorentzian frequencies of half width `width` gives
    `r' = -width r + K1 (r - r^3) / 2 + K2 (r^3 - r^5) / 2`. Without links, the incoherent state stays stable for every
    `K2`, and for `K2 > 8 width` a synchronized state of `r^2 = (1 + sqrt(1 - 8 width / K2)) / 2` coexists with it, so
    the order parameter depends on the initial phases and sweeps of the coupling show hysteresis. The system has the
    phases `oscillators.theta` and the order parameter `order`.

    Args:
        n_oscillators:
            The number of oscillators, the nodes of the hypergraph.

        K1:
            The pairwise coupling, divided by the mean number of links of a node.

        K2:
            The three-body coupling, divided by the mean number of triangles of a node.

        width:
            The half width of the Lorentzian distribution of the natural frequencies.

        links:
            The probability that a pair of oscillators is a link.

        triangles:
            The probability that a triple of oscillators is a triangle.

        phases:
            The initial phases, or `None` for phases drawn uniformly from the circle.

        seed:
            The seed of the hypergraph, the frequencies, and the phases.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import HypergraphKuramotoSystem
        >>> system = HypergraphKuramotoSystem(6).prepare().system_
        >>> len(system.elements[0].space.prepare().hyperedges_)
        35
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'oscillators', 'phase', 'hypergraph')
    references = ('Skardal and Arenas 2019', 'Majhi et al. 2022')
    until = 40.0
    dt = 0.1
    observables = tuple(order_parameter('order', 'oscillators'))

    def __init__(
        self,
        n_oscillators: int = 60,
        *,
        K1: float = 0.0,
        K2: float = 8.0,
        width: float = 0.5,
        links: float = 1.0,
        triangles: float = 1.0,
        phases: Sequence[float] | None = None,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.K1 = K1
        self.K2 = K2
        self.width = width
        self.links = links
        self.triangles = triangles
        self.phases = phases
        self.seed = seed

    @staticmethod
    def _get_simplices(n_oscillators: int, links: float, triangles: float, seed: int) -> xgi.Hypergraph:
        """Return a random hypergraph over the oscillators.

        The hypergraph has the given densities of links and of triangles.
        """
        return xgi.fast_random_hypergraph(n_oscillators, [links, triangles], seed=seed)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Kuramoto oscillators coupled in pairs and by three-body interactions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        hypergraph = self._get_simplices(self.n_oscillators, self.links, self.triangles, self.seed)
        space = NetworkSpace(nx.complete_graph(self.n_oscillators), hyperedges=hypergraph)
        sizes = [len(members) for members in hypergraph.edges.members(dtype=list)]
        mean_links = max(PAIR * sizes.count(PAIR) / self.n_oscillators, 1.0)
        mean_triangles = max(TRIANGLE * sizes.count(TRIANGLE) / self.n_oscillators, 1.0)
        rng = np.random.default_rng(self.seed + 1)
        omega = self.width * stats.cauchy.ppf(rng.uniform(size=self.n_oscillators))
        theta = (
            rng.uniform(0, 2 * math.pi, self.n_oscillators)
            if self.phases is None
            else np.asarray(self.phases, dtype=float)
        )
        coupling = SimplexCouplingInteraction(
            'coupling',
            target='oscillators',
            K1=self.K1 / mean_links,
            K2=self.K2 / mean_triangles,
        )
        self.system_ = System(
            self.label,
            elements=[Element('oscillators', values={'theta': theta, 'omega': omega}, space=space)],
            interactions=[coupling, PhaseRotationInteraction('rotation', target='oscillators')],
        )
