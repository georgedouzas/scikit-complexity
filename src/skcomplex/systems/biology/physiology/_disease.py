"""Physiology systems: the unified disease model of tissue and immune cells as adaptive networks of oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement, LatticeElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import LayerSpec
from ..._catalog import register_system
from ._adaptive_phase_coupling import AdaptivePhaseCouplingInteraction
from ._cell_oscillator import CellOscillatorElement
from ._cytokine_link import CytokineLinkElement
from ._layer_coupling import LayerCouplingInteraction


@register_system(
    field='biology/physiology',
    tags=('sepsis', 'adaptive network', 'multiplex', 'phase oscillators', 'ode'),
    references=('Sawicki et al. 2022', 'Berner et al. 2023'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def two_layer_sepsis(
    n_cells: int = 20,
    *,
    beta: float = 0.5 * math.pi,
    sigma: float = 1.0,
    pathological: float = 0.0,
    omega_healthy: float = 0.0,
    omega_pathological: float = 1.0,
    omega_immune: float = 0.0,
    alpha_p: float = 0.3 * math.pi,
    alpha_s: float = 0.35 * math.pi,
    epsilon_p: float = 0.03,
    epsilon_s: float = 0.3,
    connectivity: float = 0.5,
    seed: int | None = 0,
) -> System:
    """Return the two-layer model of tumour disease and sepsis, parenchymal and immune cells as adaptive oscillators.

    The parenchymal layer of `n_cells` phase oscillators couples through fixed links `a_ij`, drawn as 1 with the
    probability `connectivity`, and adaptive cytokine links `kappa_ij`. The immune layer couples through adaptive links
    alone, which adapt ten times faster, and every cell couples to its partner in the other layer with strength
    `sigma`. Following Eqs. (40) and (41) of Berner et al. (2023), `dphi_i/dt = omega_i - sum_j (a_ij + kappa_ij)
    sin(phi_i - phi_j + alpha) / N - sigma sin(phi_i - phi_i')` and `dkappa_ij/dt = -epsilon (kappa_ij + sin(phi_i -
    phi_j - beta))`. The healthy state is the frequency-synchronized one, and the age parameter `beta` and a fraction
    `pathological` of cells with another natural frequency push the layers into desynchronized, pathological states.

    Args:
        n_cells:
            The number of cells of each layer.

        beta:
            The age parameter of the adaptation.

        sigma:
            The coupling between the layers.

        pathological:
            The fraction of pathological parenchymal cells.

        omega_healthy:
            The natural frequency of healthy parenchymal cells.

        omega_pathological:
            The natural frequency of pathological parenchymal cells.

        omega_immune:
            The natural frequency of the immune cells.

        alpha_p:
            The phase lag of the parenchymal layer.

        alpha_s:
            The phase lag of the immune layer.

        epsilon_p:
            The adaptation rate of the parenchymal links.

        epsilon_s:
            The adaptation rate of the immune links.

        connectivity:
            The probability of a fixed parenchymal link.

        seed:
            The seed of the fixed links, the pathological cells, and the initial phases and couplings.

    Returns:
        system:
            The system, with the phases `cells.phi`, the parenchymal cells first and the immune cells after, and the
            couplings `parenchymal_links.kappa` and `immune_links.kappa`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import two_layer_sepsis
        >>> two_layer_sepsis(5).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(seed)
    n = n_cells
    parenchyma = nx.complete_graph(range(n), create_using=nx.DiGraph)
    parenchyma.add_nodes_from(range(n, 2 * n))
    immune = nx.DiGraph()
    immune.add_nodes_from(range(2 * n))
    immune.add_edges_from((i, j) for i in range(n, 2 * n) for j in range(n, 2 * n) if i != j)
    partners = nx.DiGraph()
    partners.add_nodes_from(range(2 * n))
    partners.add_edges_from([(i, i + n) for i in range(n)] + [(i + n, i) for i in range(n)])
    space = NetworkSpace(parenchyma, layers=[LayerSpec('immune', immune), LayerSpec('partners', partners)])
    frequencies = np.full(2 * n, omega_immune)
    sick = rng.random(n) < pathological
    frequencies[:n] = np.where(sick, omega_pathological, omega_healthy)
    cells = LatticeElement(
        'cells',
        element=CellOscillatorElement,
        space=space,
        values={'phi': rng.uniform(0, 2 * math.pi, 2 * n), 'omega': frequencies},
    )
    edges = n * (n - 1)
    parenchymal_links = GroupElement(
        'parenchymal_links',
        element=CytokineLinkElement,
        space=space,
        on='edges',
        values={'kappa': rng.uniform(-1, 1, edges), 'a': (rng.random(edges) < connectivity).astype(float)},
    )
    immune_links = GroupElement(
        'immune_links',
        element=CytokineLinkElement,
        space=space,
        on='edges',
        layer='immune',
        values={'kappa': rng.uniform(-1, 1, edges)},
    )
    return System(
        elements=[cells, parenchymal_links, immune_links],
        interactions=[
            AdaptivePhaseCouplingInteraction(
                'parenchymal_coupling',
                cells='cells',
                links='parenchymal_links',
                N=n,
                alpha=alpha_p,
                beta=beta,
                epsilon=epsilon_p,
            ),
            AdaptivePhaseCouplingInteraction(
                'immune_coupling',
                cells='cells',
                links='immune_links',
                N=n,
                alpha=alpha_s,
                beta=beta,
                epsilon=epsilon_s,
            ),
            LayerCouplingInteraction('interlayer', cells='cells', sigma=sigma, layer='partners'),
        ],
    )
