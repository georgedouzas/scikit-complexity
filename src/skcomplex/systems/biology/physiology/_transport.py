"""Physiology systems: vascular networks whose vessels adapt to the flows they carry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._flow_adaptation import FlowAdaptationInteraction
from ._vessel_segment import VesselSegmentElement


@register_system(
    field='biology/physiology',
    tags=('adaptive network', 'vasculature', 'kirchhoff', 'remodelling', 'ode'),
    references=('Hu and Cai 2013', 'Martens and Klemm 2017', 'Berner et al. 2023'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def adaptive_flow_network(
    side: int = 6,
    *,
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    gamma: float = 2.0,
    inflow: float | None = None,
    seed: int | None = 0,
) -> System:
    """Return a square grid of vessels fed at one corner and drained at every other node, whose conductances adapt.

    The pressures balance the flows at every node, `h_i = sum_j C_ij (p_i - p_j)`, and every conductance grows with the
    dissipated power and decays, `dC_ij/dt = C_ij (alpha1 |p_i - p_j|^gamma - alpha2)`. Under a steady supply the
    network settles where every surviving vessel carries the same pressure drop `(alpha2 / alpha1)^(1 / gamma)`, so the
    conductances grow with the flow each vessel must carry: the vessels near the source thicken into trunks and the
    distal ones stay thin.

    Args:
        side:
            The number of nodes along each side of the grid.

        alpha1:
            The growth of the conductance with the pressure drop.

        alpha2:
            The decay rate of the conductance.

        gamma:
            The exponent of the pressure drop.

        inflow:
            The flow into the source at the first corner, which the other nodes drain equally, or `None` for the number
            of edges.

        seed:
            The seed of the initial conductances.

    Returns:
        system:
            The system, with the conductances `vessels.C` on the edges of the grid.

    Examples:
        >>> from skcomplex.systems.biology.physiology import adaptive_flow_network
        >>> adaptive_flow_network(3).elements[0].prepare().size_
        12
    """
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    space = NetworkSpace(graph).prepare()
    sources, targets = space.edges_[None]
    n_nodes, n_edges = graph.number_of_nodes(), len(sources)
    supply = float(n_edges) if inflow is None else inflow
    h = np.full(n_nodes, -supply / (n_nodes - 1))
    h[0] = supply
    rng = np.random.default_rng(seed)
    vessels = GroupElement(
        'vessels',
        element=VesselSegmentElement,
        space=space,
        on='edges',
        values={'C': rng.uniform(0.5, 1.5, n_edges)},
    )
    adaptation = FlowAdaptationInteraction(
        'adaptation',
        vessels='vessels',
        edges=tuple(zip(sources.tolist(), targets.tolist(), strict=True)),
        h=tuple(h.tolist()),
        alpha1=alpha1,
        alpha2=alpha2,
        gamma=gamma,
    )
    return System(elements=[vessels], interactions=[adaptation])
