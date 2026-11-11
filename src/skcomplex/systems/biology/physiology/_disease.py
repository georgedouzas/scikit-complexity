"""The unified disease model of tissue and immune cells.

The cells form adaptive networks of oscillators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import LayerSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class CellOscillatorElement(Element):
    """Functional unit of a tissue layer, a cell oscillator `phi' = omega`.

    The unit is reduced to a phase.

    Healthy and pathological cells differ by their metabolism, which the natural frequency `omega` stands for.
    Couplings to other cells add to the rate of the phase.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phi:
            The initial phase.

        omega:
            The natural frequency.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CellOscillatorElement
        >>> CellOscillatorElement('cell', omega=0.2).omega
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rotation of the phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.phi, s.omega)]


class CytokineLinkElement(Element):
    """Link between two cells of a layer.

    It has a fixed connectivity `a` and an adaptive cytokine coupling `kappa`.

    Args:
        label:
            The label of the link or of the group on the edges.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        kappa:
            The initial adaptive coupling, between -1 and 1.

        a:
            The fixed connectivity, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CytokineLinkElement
        >>> CytokineLinkElement('link', a=1.0).a
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec(domain=RealSet.closed(-1, 1))] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa
        self.a = a


class AdaptivePhaseCouplingInteraction(NeighborInteraction):
    """Phase coupling of the cells of a layer through adaptive links.

    The links adapt to the phase differences they carry.

    Along the link from cell `j` to cell `i`, `dphi_i/dt += -(a_ij + kappa_ij) sin(phi_i - phi_j + alpha) / N` and
    `dkappa_ij/dt = -epsilon (kappa_ij + sin(phi_i - phi_j - beta))`. The phase lag `alpha` sets the preferred phase
    difference of the coupling, and the age parameter `beta` the rule of the adaptation.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells on the nodes.

        links:
            The path of the group of links on the edges of the layer.

        N:
            The number of cells of the layer, which normalizes the coupling.

        alpha:
            The phase lag of the coupling.

        beta:
            The age parameter of the adaptation.

        epsilon:
            The adaptation rate.

    Examples:
        >>> from skcomplex.systems.biology.physiology import AdaptivePhaseCouplingInteraction
        >>> AdaptivePhaseCouplingInteraction('coupling', cells='cells', links='links', epsilon=0.3).epsilon
        0.3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        N: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.cells = cells
        self.links = links
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the coupling of the receiving cell and the adaptation of the link.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        difference = s.target.phi - s.source.phi
        kappa = s.edge.kappa
        return [
            RateLaw(s.target.phi, -(s.edge.a + kappa) * sin(difference + s.alpha) / s.N),
            RateLaw(kappa, -s.epsilon * (kappa + sin(difference - s.beta))),
        ]


class LayerCouplingInteraction(NeighborInteraction):
    """Coupling of every cell to its partner in another layer.

    It adds `dphi_i/dt += -sigma sin(phi_i - phi_j)`.

    It runs on the edges of the layer of the network that joins the partners, both ways.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells on the nodes.

        sigma:
            The strength of the coupling between the layers.

        layer:
            The layer of the network whose edges join the partners.

    Examples:
        >>> from skcomplex.systems.biology.physiology import LayerCouplingInteraction
        >>> LayerCouplingInteraction('interlayer', cells='cells', sigma=0.5, layer='partners').layer
        'partners'
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.cells = cells
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the pull of the receiving cell towards its partner.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.phi, -s.sigma * sin(s.target.phi - s.source.phi))]


class TwoLayerSepsisSystem(BaseSystem):
    """The two-layer model of tumour disease and sepsis.

    Its parenchymal and immune cells are adaptive oscillators.

    The parenchymal layer of `n_cells` phase oscillators couples through fixed links `a_ij`, drawn as 1 with the
    probability `connectivity`, and adaptive cytokine links `kappa_ij`. The immune layer couples through adaptive links
    alone, which adapt ten times faster, and every cell couples to its partner in the other layer with strength `sigma`.
    Following Eqs. (40) and (41) of Berner et al. (2023), `dphi_i/dt = omega_i - sum_j (a_ij + kappa_ij) sin(phi_i -
    phi_j + alpha) / N - sigma sin(phi_i - phi_i')` and `dkappa_ij/dt = -epsilon (kappa_ij + sin(phi_i - phi_j -
    beta))`. The healthy state is the frequency-synchronized one, and the age parameter `beta` and a fraction
    `pathological` of cells with another natural frequency push the layers into desynchronized, pathological states. The
    system has the phases `cells.phi`, the parenchymal cells first and the immune cells after, and the couplings
    `parenchymal_links.kappa` and `immune_links.kappa`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import TwoLayerSepsisSystem
        >>> TwoLayerSepsisSystem(5).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'biology/physiology'
    topics = ('sepsis', 'adaptive network', 'multiplex', 'phase oscillators', 'ode')
    references = ('Sawicki et al. 2022', 'Berner et al. 2023')
    until = 200.0
    dt = 1.0

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_cells = n_cells
        self.beta = beta
        self.sigma = sigma
        self.pathological = pathological
        self.omega_healthy = omega_healthy
        self.omega_pathological = omega_pathological
        self.omega_immune = omega_immune
        self.alpha_p = alpha_p
        self.alpha_s = alpha_s
        self.epsilon_p = epsilon_p
        self.epsilon_s = epsilon_s
        self.connectivity = connectivity
        self.seed = seed

    def _build_system(self) -> CompositeSystem:
        """Return the components of the two-layer model of tumour disease and sepsis."""
        rng = np.random.default_rng(self.seed)
        n = self.n_cells
        parenchyma = nx.complete_graph(range(n), create_using=nx.DiGraph)
        parenchyma.add_nodes_from(range(n, 2 * n))
        immune = nx.DiGraph()
        immune.add_nodes_from(range(2 * n))
        immune.add_edges_from((i, j) for i in range(n, 2 * n) for j in range(n, 2 * n) if i != j)
        partners = nx.DiGraph()
        partners.add_nodes_from(range(2 * n))
        partners.add_edges_from([(i, i + n) for i in range(n)] + [(i + n, i) for i in range(n)])
        space = NetworkSpace(parenchyma, layers=[LayerSpec('immune', immune), LayerSpec('partners', partners)])
        frequencies = np.full(2 * n, self.omega_immune)
        sick = rng.random(n) < self.pathological
        frequencies[:n] = np.where(sick, self.omega_pathological, self.omega_healthy)
        cells = GroupElement(
            'cells',
            element=CellOscillatorElement,
            space=space,
            values={'phi': rng.uniform(0, 2 * math.pi, 2 * n), 'omega': frequencies},
            on='sites',
        )
        edges = n * (n - 1)
        parenchymal_links = GroupElement(
            'parenchymal_links',
            element=CytokineLinkElement,
            space=space,
            on='edges',
            values={'kappa': rng.uniform(-1, 1, edges), 'a': (rng.random(edges) < self.connectivity).astype(float)},
        )
        immune_links = GroupElement(
            'immune_links',
            element=CytokineLinkElement,
            space=space,
            on='edges',
            layer='immune',
            values={'kappa': rng.uniform(-1, 1, edges)},
        )
        return CompositeSystem(
            self.label,
            elements=[cells, parenchymal_links, immune_links],
            interactions=[
                AdaptivePhaseCouplingInteraction(
                    'parenchymal_coupling',
                    cells='cells',
                    links='parenchymal_links',
                    N=n,
                    alpha=self.alpha_p,
                    beta=self.beta,
                    epsilon=self.epsilon_p,
                ),
                AdaptivePhaseCouplingInteraction(
                    'immune_coupling',
                    cells='cells',
                    links='immune_links',
                    N=n,
                    alpha=self.alpha_s,
                    beta=self.beta,
                    epsilon=self.epsilon_s,
                ),
                LayerCouplingInteraction('interlayer', cells='cells', sigma=self.sigma, layer='partners'),
            ],
        )
