"""Topological synchronization.

Phases on the nodes and links of a clique complex are coupled by order parameters.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import Expression, cos, sin, sqrt

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import codifferential, exterior_derivative, mean
from ....core.components.spaces import BaseSpace, CellComplexSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ..common.oscillators import LinkElement, PhaseOscillatorElement, PhaseRotationInteraction

MODELS = ('NL', 'NLT')


class LinkOscillatorElement(PhaseOscillatorElement):
    """Phase oscillator on a link, drawn as a cell coloured by its phase."""

    visual = VisualSpec('cell', {'state': 'theta'})


class NodeCouplingInteraction(Interaction):
    """Coupling of the phases on the nodes of a cell complex through its links.

    The node phases turn by `K R_links codifferential(sin(d theta))`, where `R_links` is the order parameter of the
    projection `codifferential(phi)` of the link phases on the nodes, as in the topological synchronization of
    Ghorbanchian et al.

    Args:
        label:
            The label of the interaction.

        nodes:
            The path of the phases on the nodes.

        links:
            The path of the phases on the links.

        K:
            The coupling.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import NodeCouplingInteraction
        >>> NodeCouplingInteraction('coupling', nodes='nodes', links='links', K=2.0).K
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, nodes: Role, links: Role, K: float = 1.0) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.K = K

    @staticmethod
    def _get_order_of(expression: Expression, over: SimpleNamespace) -> Expression:
        """Return the modulus of the mean unit vector of the angles of an expression over a group."""
        return sqrt(mean(cos(expression), over=over) ** 2 + mean(sin(expression), over=over) ** 2)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the node phases.

        Args:
            context:
                The namespace of the nodes, the links, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            nodes, links = context.nodes, context.links
            order = NodeCouplingInteraction._get_order_of(codifferential(links.theta), nodes)
            rate = context.K * order * codifferential(sin(exterior_derivative(nodes.theta)))
            self.laws_ = [RateLaw(nodes.theta.diff(context.t), rate)]


class LinkCouplingInteraction(Interaction):
    """Coupling of the phases on the links of a cell complex through its nodes and its triangles.

    The link phases turn by `K (R_nodes d(sin(codifferential(phi))) - codifferential(sin(d phi)))`, Eq. 6 of the review
    of Majhi, Perc, and Ghosh. With triangles, Eq. 8, the coupling through the nodes is scaled by `R_nodes R_up` and the
    coupling through the triangles by `R_links`, where `R_up` is the order parameter of the projection `d phi` of the
    link phases on the triangles.

    Args:
        label:
            The label of the interaction.

        links:
            The path of the phases on the links.

        nodes:
            The path of the phases on the nodes.

        triangles:
            The path of the element on the triangles, or `None` for the coupling of Eq. 6.

        K:
            The coupling.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LinkCouplingInteraction
        >>> LinkCouplingInteraction('coupling', links='links', nodes='nodes').triangles is None
        True
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        links: Role,
        nodes: Role,
        triangles: Role | None = None,
        K: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.links = links
        self.nodes = nodes
        self.triangles = triangles
        self.K = K

    @staticmethod
    def _get_order_of(expression: Expression, over: SimpleNamespace) -> Expression:
        """Return the modulus of the mean unit vector of the angles of an expression over a group."""
        return sqrt(mean(cos(expression), over=over) ** 2 + mean(sin(expression), over=over) ** 2)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the link phases.

        Args:
            context:
                The namespace of the links, the nodes, the triangles, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            links, nodes = context.links, context.nodes
            down = exterior_derivative(sin(codifferential(links.theta)))
            up = codifferential(sin(exterior_derivative(links.theta)))
            node_order = LinkCouplingInteraction._get_order_of(nodes.theta, nodes)
            if self.triangles is None:
                rate = context.K * (node_order * down - up)
            else:
                order_down = LinkCouplingInteraction._get_order_of(codifferential(links.theta), nodes)
                order_up = LinkCouplingInteraction._get_order_of(exterior_derivative(links.theta), context.triangles)
                rate = context.K * (node_order * order_up * down - order_down * up)
            self.laws_ = [RateLaw(links.theta.diff(context.t), rate)]


class TopologicalKuramotoSystem(BaseSystem):
    """Phases on the nodes and links of a clique complex.

    The phases are coupled through its incidence by order parameters.

    Node phases follow `theta' = omega + K R_links codifferential(sin(d theta))`. Link phases follow `phi' = omega + K
    R_nodes d(sin(codifferential(phi))) - K codifferential(sin(d phi))`. With the signs of `codifferential`, these are
    the Nodes-Links system of Ghorbanchian et al. (2021), Eq. 6 of the review of Majhi, Perc, and Ghosh (2022,
    arXiv 2203.06601 §III), which is the source at hand. Here `R_nodes` is the order parameter of the node phases and
    `R_links` that of the projection of the link phases on the nodes. The review takes frequencies of mean `Omega` and
    standard deviation `1 / tau`, with `Omega = 2` and `tau = 1` for its Fig. 2, on complexes of 500 nodes. The mean 0,
    the size, and the random graph are choices of this system.

    The Nodes-Links-Triangles system, `model='NLT'`, Eq. 8 of the review, scales the coupling of the link phases through
    the nodes by `R_nodes R_up` and through the triangles by `R_links`, where `R_up` is the order parameter of the
    projection of the link phases on the triangles. Every projection then couples to another, so all three order
    parameters jump together at one explosive transition, while in the Nodes-Links system `R_up` moves on its own. The
    system has a group `triangles` of unit weights on the faces for the Nodes-Links-Triangles system, which only gives
    the order parameter of the projection on the triangles.

    Args:
        n_nodes:
            The number of nodes of the random graph.

        density:
            The probability of every edge of the random graph.

        coupling:
            The coupling `K`.

        width:
            The standard deviation of the natural frequencies of the nodes and the links.

        seed:
            The seed of the graph, the frequencies, and the initial phases.

        model:
            The coupling, `'NL'` for the Nodes-Links system or `'NLT'` for the Nodes-Links-Triangles system.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The model is neither `'NL'` nor `'NLT'`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import TopologicalKuramotoSystem
        >>> system = TopologicalKuramotoSystem(6, density=0.8).prepare().system_
        >>> [len(element.theta) for element in system.elements]
        [6, 13]
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'oscillators', 'phase', 'simplicial complex', 'cell complex', 'explosive')
    references = ('Majhi, Perc, and Ghosh 2022', 'Ghorbanchian et al. 2021')
    until = 40.0
    dt = 0.2

    @staticmethod
    def _get_modulus(values: object) -> float:
        """Return the modulus of the mean of the unit vectors of some angles.

        The modulus is 0 without angles.
        """
        angles = np.ravel(np.asarray(values, dtype=float))
        return float(np.abs(np.exp(1j * angles).mean())) if angles.size else 0.0

    @staticmethod
    def _get_node_order(t: float, values: Mapping[str, object]) -> float:
        """Return the order parameter of the node phases."""
        del t
        return TopologicalKuramotoSystem._get_modulus(values['nodes.theta'])

    @staticmethod
    def _link_order(
        edges: scipy.sparse.csr_array,
        faces: scipy.sparse.csr_array,
        t: float,
        values: Mapping[str, object],
    ) -> float:
        """Return the mean order parameter of the projections of the link phases.

        The projections are those on the nodes and on the triangles.
        """
        del t
        phases = np.ravel(np.asarray(values['links.theta'], dtype=float))
        return (
            TopologicalKuramotoSystem._get_modulus(edges.T @ phases)
            + TopologicalKuramotoSystem._get_modulus(faces @ phases)
        ) / 2

    def __init__(
        self,
        n_nodes: int = 30,
        *,
        density: float = 0.4,
        coupling: float = 2.0,
        width: float = 1.0,
        seed: int = 0,
        model: str = 'NL',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.density = density
        self.coupling = coupling
        self.width = width
        self.seed = seed
        self.model = model

    @staticmethod
    def _get_topological_observables(space: CellComplexSpace) -> list[Observable]:
        """Return the order parameters of the node phases and the link phase projections."""
        return [
            Observable('node_order', function=TopologicalKuramotoSystem._get_node_order),
            Observable(
                'link_order',
                function=partial(TopologicalKuramotoSystem._link_order, *space.incidences_[:2]),
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the phases on the nodes and links of a clique complex of a random graph.

        Args:
            context:
                The observables to compile with the problem, or `None` for the order parameters.

        Raises:
            ValidationError:
                The model is neither `'NL'` nor `'NLT'`.
        """
        super()._prepare(context)
        if self.model not in MODELS:
            error_msg = f'Unknown model {self.model!r}. Use one of {", ".join(MODELS)}.'
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.seed)
        graph = nx.gnp_random_graph(self.n_nodes, self.density, seed=self.seed)
        space = CellComplexSpace(NetworkSpace(graph), faces='cliques').prepare()
        n_links = space.incidences_[0].shape[0]
        nodes = PhaseOscillatorElement(
            'nodes',
            theta=rng.uniform(-np.pi, np.pi, self.n_nodes),
            omega=rng.normal(0.0, self.width, self.n_nodes),
            space=space,
        )
        links = LinkOscillatorElement(
            'links',
            theta=rng.uniform(-np.pi, np.pi, n_links),
            omega=rng.normal(0.0, self.width, n_links),
            space=space,
            on='edges',
        )
        elements: list[Element] = [nodes, links]
        triangles = None
        if self.model == 'NLT':
            elements.append(LinkElement('triangles', space=space, on='hyperedges'))
            triangles = 'triangles'
        interactions = [
            PhaseRotationInteraction('node_rotation', target='nodes'),
            PhaseRotationInteraction('link_rotation', target='links'),
            NodeCouplingInteraction('node_coupling', nodes='nodes', links='links', K=self.coupling),
            LinkCouplingInteraction(
                'link_coupling',
                links='links',
                nodes='nodes',
                triangles=triangles,
                K=self.coupling,
            ),
        ]
        if not context:
            self.observables_ = tuple(self._get_topological_observables(space))
        self.system_ = System(self.label, elements=elements, interactions=interactions)
