"""Intercellular exchange: substances passed between cells in a closed contour and on networks of junctions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np

from ....core.elements import FieldElement, GroupElement
from ....core.spaces import ContinuousTime, DiscreteTime, NetworkSpace
from ....core.system import System
from ..._catalog import register_model
from ._contour_cell import ContourCellElement
from ._contour_uptake import ContourUptakeInteraction
from ._exchanging_cell import ExchangingCellElement


@register_model(
    field='biology/physiology',
    tags=('map', 'logistic', 'coupled maps', 'ring', 'synchronization', 'cell communication'),
    references=('Mihailović et al. 2023',),
    defaults={'until': 200, 'dt': 1},
)
def closed_contour_cell_exchange(
    n_cells: int = 8,
    *,
    r: float = 2.5,
    c: float | Sequence[float] = 0.1,
    p: float | Sequence[float] = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the exchange of signalling molecules between logistic cells arranged in a closed contour.

    Every cell runs the logistic map on its concentration and takes up a share `c_i` of its next state from the next
    cell through the power map `x_{i+1}^{p_i}`, `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^{p_i}`. Mihailović et
    al. show that an equilibrium is asymptotically stable for every coupling when every `x_i` lies in
    `((r - 1) / 2r, (r + 1) / 2r)` and above `p_{i-1}^(1 / (1 - p_{i-1}))`. In this domain the cells synchronize to a
    common concentration, and a chaotic environment, `r` near 4, desynchronizes them.

    Args:
        n_cells:
            The number of cells of the contour.

        r:
            The logistic parameter of every cell.

        c:
            The coupling of every cell to the next, one value or one per cell.

        p:
            The affinity of every cell for the uptake, one value or one per cell.

        random_state:
            The seed or generator of the initial concentrations, uniform between 0.2 and 0.8.

    Returns:
        system:
            The model, with the concentrations `cells.x`, cell `i + 1` next to cell `i`.

    Examples:
        >>> from skcomplex.models.biology.physiology import closed_contour_cell_exchange
        >>> closed_contour_cell_exchange(5).elements[0].check().prepare().size_
        5
    """
    rng = np.random.default_rng(random_state)
    cells = GroupElement(
        'cells',
        element=ContourCellElement,
        size=n_cells,
        values={
            'x': rng.uniform(0.2, 0.8, n_cells),
            'r': r,
            'c': np.broadcast_to(np.asarray(c, dtype=float), n_cells).copy(),
            'p': np.broadcast_to(np.asarray(p, dtype=float), n_cells).copy(),
        },
    )
    uptake = ContourUptakeInteraction('uptake', target='cells')
    return System(elements=[cells], interactions=[uptake], time=DiscreteTime())


@register_model(
    field='biology/physiology',
    tags=('network', 'cell communication', 'gap junctions', 'nanotubes', 'stability', 'ode'),
    references=('Mihailović et al. 2023',),
    defaults={'until': 100.0, 'dt': 0.5},
)
def nanotube_gap_junction_exchange(
    graph: nx.Graph | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 10.0,
    g: float = 0.1,
    xi: float = 0.0,
    x: float | Sequence[float] | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the intercellular exchange of a substance through gap junctions and tunnelling nanotubes on a network.

    Every cell metabolizes the substance at the Michaelis-Menten rate `alpha x / (beta + x)` and exchanges it with
    its neighbours, by diffusion `g (x_j - x_i)` through gap junctions and by the transfer `xi x_j` through nanotubes.
    The state without substance is stable while the largest eigenvalue of `xi A - g L` stays below `alpha / beta`,
    where `A` is the adjacency and `L` the Laplacian of the network. Past the threshold `xi_crit` the nanotubes
    destabilize the communication and the concentrations grow. On a regular graph of degree `k`,
    `xi_crit = alpha / (beta k)`.

    Args:
        graph:
            The NetworkX graph of the gap junctions, or `None` for a modular Erdős-Rényi network of 100 cells in ten
            modules, with an attachment probability of 0.03 overall and 90% of the links within modules.

        alpha:
            The maximal rate of metabolism.

        beta:
            The half-saturation concentration of metabolism.

        g:
            The exchange coefficient of the gap junctions.

        xi:
            The strength of the nanotubes.

        x:
            The initial concentrations, one value or one per node in node order, or `None` for values drawn uniformly
            between 0 and 1.

        random_state:
            The seed or generator of the default network and of the initial concentrations.

    Returns:
        system:
            The model, with the concentrations `cells.x`, one value per node.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.biology.physiology import nanotube_gap_junction_exchange
        >>> nanotube_gap_junction_exchange(nx.cycle_graph(6), xi=0.01).elements[0].check().prepare().size_
        6
    """
    rng = np.random.default_rng(random_state)
    if graph is None:
        graph = nx.random_partition_graph([10] * 10, 0.297, 0.0033, seed=int(rng.integers(2**31)))
    size = graph.number_of_nodes()
    initial = rng.uniform(0.0, 1.0, size) if x is None else np.broadcast_to(np.asarray(x, dtype=float), size).copy()
    values = {
        'x': initial,
        'alpha': alpha,
        'beta': beta,
        'g': g,
        'xi': xi,
        'degree': np.array([graph.degree[node] for node in graph.nodes], dtype=float),
    }
    cells = FieldElement('cells', element=ExchangingCellElement, space=NetworkSpace(graph), values=values)
    return System(elements=[cells], time=ContinuousTime())
