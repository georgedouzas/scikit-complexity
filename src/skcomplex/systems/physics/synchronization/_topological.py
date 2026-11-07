"""Topological synchronization: phases on the nodes and links of a clique complex, coupled by order parameters."""

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
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, VisualSpec
from ..._catalog import register_system
from ..common.oscillators import LinkElement, PhaseOscillatorElement

MODELS = ('NL', 'NLT')


def _order_of(expression: Expression, over: SimpleNamespace) -> Expression:
    """Return the modulus of the mean of the unit vectors of the angles of an expression over a group."""
    return sqrt(mean(cos(expression), over=over) ** 2 + mean(sin(expression), over=over) ** 2)


def _node_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
    """Return the coupling of a node phase, scaled by the order parameter of the projections of the link phases."""
    nodes, links = namespace.target, namespace.sources[0]
    order = _order_of(codifferential(links.theta), nodes)
    return coupling * order * codifferential(sin(exterior_derivative(nodes.theta)))


def _link_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
    """Return the coupling of a link phase through its nodes, scaled by the order of the nodes, and its triangles."""
    links, nodes = namespace.target, namespace.sources[0]
    down = exterior_derivative(sin(codifferential(links.theta)))
    return coupling * (_order_of(nodes.theta, nodes) * down - codifferential(sin(exterior_derivative(links.theta))))


def _triangle_link_rate(coupling: float, namespace: SimpleNamespace) -> Expression:
    """Return the link coupling of the Nodes-Links-Triangles system, each term scaled by the order of the other face."""
    links, nodes, triangles = namespace.target, namespace.sources[0], namespace.sources[1]
    down = exterior_derivative(sin(codifferential(links.theta)))
    up = codifferential(sin(exterior_derivative(links.theta)))
    order_down = _order_of(codifferential(links.theta), nodes)
    order_up = _order_of(exterior_derivative(links.theta), triangles)
    return coupling * (_order_of(nodes.theta, nodes) * order_up * down - order_down * up)


def _modulus(values: object) -> float:
    """Return the modulus of the mean of the unit vectors of some angles, 0 without angles."""
    angles = np.ravel(np.asarray(values, dtype=float))
    return float(np.abs(np.exp(1j * angles).mean())) if angles.size else 0.0


def _node_order(t: float, values: Mapping[str, object]) -> float:
    """Return the order parameter of the node phases."""
    del t
    return _modulus(values['nodes.theta'])


def _link_order(
    edges: scipy.sparse.csr_array,
    faces: scipy.sparse.csr_array,
    t: float,
    values: Mapping[str, object],
) -> float:
    """Return the mean order parameter of the projections of the link phases on the nodes and on the triangles."""
    del t
    phases = np.ravel(np.asarray(values['links.theta'], dtype=float))
    return (_modulus(edges.T @ phases) + _modulus(faces @ phases)) / 2


def _topological_observables(system: System) -> list[ObservableSpec]:
    """Return the order parameters of the node phases and of the projections of the link phases."""
    space = system.elements[0].space.prepare()
    return [
        ObservableSpec('node_order', function=_node_order),
        ObservableSpec('link_order', function=partial(_link_order, *space.incidences_[:2])),
    ]


@register_system(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'simplicial complex', 'cell complex', 'explosive'),
    references=('Majhi, Perc, and Ghosh 2022', 'Ghorbanchian et al. 2021'),
    defaults={'until': 40.0, 'dt': 0.2},
    observables=_topological_observables,
)
def topological_kuramoto(
    n_nodes: int = 30,
    *,
    density: float = 0.4,
    coupling: float = 2.0,
    width: float = 1.0,
    seed: int = 0,
    model: str = 'NL',
) -> System:
    """Return phases on the nodes and links of a clique complex, coupled through its incidence by order parameters.

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
    parameters jump together at one explosive transition, while in the Nodes-Links system `R_up` moves on its own.

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

    Returns:
        system:
            The system, with a group `triangles` of unit weights on the faces for the Nodes-Links-Triangles system,
            which only gives the order parameter of the projection on the triangles.

    Raises:
        ValidationError:
            The model is neither `'NL'` nor `'NLT'`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import topological_kuramoto
        >>> [element.size_ for element in topological_kuramoto(6, density=0.8).prepare().elements]
        [6, 13]
    """
    if model not in MODELS:
        error_msg = f'Unknown model {model!r}. Use one of {", ".join(MODELS)}.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(n_nodes, density, seed=seed)
    space = CellComplexSpace(NetworkSpace(graph), faces='cliques').prepare()
    n_links = space.incidences_[0].shape[0]
    nodes = GroupElement(
        'nodes',
        element=PhaseOscillatorElement,
        space=space,
        on='sites',
        values={'theta': rng.uniform(-np.pi, np.pi, n_nodes), 'omega': rng.normal(0.0, width, n_nodes)},
    )
    links = GroupElement(
        'links',
        element=PhaseOscillatorElement,
        space=space,
        on='edges',
        values={'theta': rng.uniform(-np.pi, np.pi, n_links), 'omega': rng.normal(0.0, width, n_links)},
        visual=VisualSpec('cell', {'state': 'theta'}),
    )
    interactions = [
        RateInteraction(
            'node_coupling',
            target='nodes',
            state='theta',
            sources=['links'],
            rate=partial(_node_rate, coupling),
        ),
    ]
    if model == 'NL':
        link_coupling = RateInteraction(
            'link_coupling',
            target='links',
            state='theta',
            sources=['nodes'],
            rate=partial(_link_rate, coupling),
        )
        return System(elements=[nodes, links], interactions=[*interactions, link_coupling])
    triangles = GroupElement('triangles', element=LinkElement, space=space, on='hyperedges')
    link_coupling = RateInteraction(
        'link_coupling',
        target='links',
        state='theta',
        sources=['nodes', 'triangles'],
        rate=partial(_triangle_link_rate, coupling),
    )
    return System(elements=[nodes, links, triangles], interactions=[*interactions, link_coupling])
