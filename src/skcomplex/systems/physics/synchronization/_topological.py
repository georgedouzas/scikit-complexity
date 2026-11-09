"""Topological synchronization.

Phases on the nodes and links of a clique complex are coupled by order parameters.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import Expression, cos, sin, sqrt

from ....core.components.elements import GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import codifferential, exterior_derivative, mean
from ....core.components.spaces import CellComplexSpace, NetworkSpace
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, VisualSpec
from ..common.oscillators import LinkElement, PhaseOscillatorElement

MODELS = ("NL", "NLT")


def _order_of(expression: Expression, over: SimpleNamespace) -> Expression:
    """Return the modulus of the mean unit vector of the angles of an expression.

    The mean runs over a group.
    """
    return sqrt(mean(cos(expression), over=over) ** 2 + mean(sin(expression), over=over) ** 2)


def _modulus(values: object) -> float:
    """Return the modulus of the mean of the unit vectors of some angles.

    The modulus is 0 without angles.
    """
    angles = np.ravel(np.asarray(values, dtype=float))
    return float(np.abs(np.exp(1j * angles).mean())) if angles.size else 0.0


def _node_order(t: float, values: Mapping[str, object]) -> float:
    """Return the order parameter of the node phases."""
    del t
    return _modulus(values["nodes.theta"])


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
    phases = np.ravel(np.asarray(values["links.theta"], dtype=float))
    return (_modulus(edges.T @ phases) + _modulus(faces @ phases)) / 2


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

    Raises:
        ValidationError:
            The model is neither `'NL'` nor `'NLT'`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import TopologicalKuramotoSystem
        >>> system = TopologicalKuramotoSystem(6, density=0.8).prepare().system_
        >>> [element.size_ for element in system.elements]
        [6, 13]
    """

    field = "physics/synchronization"
    topics = ("synchronization", "oscillators", "phase", "simplicial complex", "cell complex", "explosive")
    references = ("Majhi, Perc, and Ghosh 2022", "Ghorbanchian et al. 2021")
    until = 40.0
    dt = 0.2

    def __init__(
        self,
        n_nodes: int = 30,
        *,
        density: float = 0.4,
        coupling: float = 2.0,
        width: float = 1.0,
        seed: int = 0,
        model: str = "NL",
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.density = density
        self.coupling = coupling
        self.width = width
        self.seed = seed
        self.model = model

    @staticmethod
    def _node_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
        """Return the coupling of a node phase.

        The coupling is scaled by the order parameter of the projections of the link phases.
        """
        nodes, links = namespace.target, namespace.sources[0]
        order = _order_of(codifferential(links.theta), nodes)
        return coupling * order * codifferential(sin(exterior_derivative(nodes.theta)))

    @staticmethod
    def _link_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
        """Return the coupling of a link phase through its nodes.

        The coupling is scaled by the order of the nodes and triangles.
        """
        links, nodes = namespace.target, namespace.sources[0]
        down = exterior_derivative(sin(codifferential(links.theta)))
        return coupling * (_order_of(nodes.theta, nodes) * down - codifferential(sin(exterior_derivative(links.theta))))

    @staticmethod
    def _triangle_link_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
        """Return the link coupling of the Nodes-Links-Triangles system.

        Each term is scaled by the other face.
        """
        links, nodes, triangles = namespace.target, namespace.sources[0], namespace.sources[1]
        down = exterior_derivative(sin(codifferential(links.theta)))
        up = codifferential(sin(exterior_derivative(links.theta)))
        order_down = _order_of(codifferential(links.theta), nodes)
        order_up = _order_of(exterior_derivative(links.theta), triangles)
        return coupling * (_order_of(nodes.theta, nodes) * order_up * down - order_down * up)

    @staticmethod
    def _topological_observables(system: System) -> list[ObservableSpec]:
        """Return the order parameters of the node phases and the link phase projections."""
        space = system.elements[0].space.prepare()
        return [
            ObservableSpec("node_order", function=_node_order),
            ObservableSpec("link_order", function=partial(_link_order, *space.incidences_[:2])),
        ]

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings["observables"] = tuple(self._topological_observables(self._build_system()))
        return settings

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.model not in MODELS:
            error_msg = f"Unknown model {self.model!r}. Use one of {', '.join(MODELS)}."
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.seed)
        graph = nx.gnp_random_graph(self.n_nodes, self.density, seed=self.seed)
        space = CellComplexSpace(NetworkSpace(graph), faces="cliques").prepare()
        n_links = space.incidences_[0].shape[0]
        nodes = GroupElement(
            "nodes",
            element=PhaseOscillatorElement,
            space=space,
            on="sites",
            values={
                "theta": rng.uniform(-np.pi, np.pi, self.n_nodes),
                "omega": rng.normal(0.0, self.width, self.n_nodes),
            },
        )
        links = GroupElement(
            "links",
            element=PhaseOscillatorElement,
            space=space,
            on="edges",
            values={"theta": rng.uniform(-np.pi, np.pi, n_links), "omega": rng.normal(0.0, self.width, n_links)},
            visual=VisualSpec("cell", {"state": "theta"}),
        )
        interactions = [
            RateInteraction(
                "node_coupling",
                target="nodes",
                state="theta",
                sources=["links"],
                rate=partial(self._node_rate, self.coupling),
            ),
        ]
        if self.model == "NL":
            link_coupling = RateInteraction(
                "link_coupling",
                target="links",
                state="theta",
                sources=["nodes"],
                rate=partial(self._link_rate, self.coupling),
            )
            return System(self.label, elements=[nodes, links], interactions=[*interactions, link_coupling])
        triangles = GroupElement("triangles", element=LinkElement, space=space, on="hyperedges")
        link_coupling = RateInteraction(
            "link_coupling",
            target="links",
            state="theta",
            sources=["nodes", "triangles"],
            rate=partial(self._triangle_link_rate, self.coupling),
        )
        return System(self.label, elements=[nodes, links, triangles], interactions=[*interactions, link_coupling])
