"""Vascular networks whose vessels adapt to the flows they carry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import CallSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class VesselSegmentElement(Element):
    """Vessel segment on an edge of a transport network.

    Its conductance adapts to the flow it carries.

    Args:
        label:
            The label of the segment or of the group on the edges.

        visual:
            How the segment is drawn, or `None` for `default_visual_`.

        C:
            The initial conductance.

    Examples:
        >>> from skcomplex.systems.biology.physiology import VesselSegmentElement
        >>> VesselSegmentElement('vessel', C=2.0).C
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        C: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.C = C


class FlowAdaptationInteraction(Interaction):
    """Adaptation of the conductances of a transport network to its pressure drops.

    The pressure drops are those of its Kirchhoff flows.

    The nodes accumulate no fluid, so the pressures `p` solve the mass balance `h_i = sum_j C_ij (p_i - p_j)` with
    the external flows `h`, which sum to zero, at every instant. Every conductance follows `dC_ij/dt = C_ij (alpha1
    |p_i - p_j|^gamma - alpha2)`: it grows with the power dissipated along its edge and decays at a constant rate. The
    balance is solved numerically with the pressure of the first node fixed at zero.

    Args:
        label:
            The label of the interaction.

        vessels:
            The path of the group of vessel segments on the edges.

        edges:
            The ends of every edge, in the order of the group.

        h:
            The external flow of every node, positive for sources and negative for sinks.

        alpha1:
            The growth of the conductance with the pressure drop.

        alpha2:
            The decay rate of the conductance.

        gamma:
            The exponent of the pressure drop.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FlowAdaptationInteraction
        >>> FlowAdaptationInteraction('adaptation', vessels='vessels', edges=((0, 1),), h=(1.0, -1.0)).gamma
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        vessels: Annotated[str, RoleSpec()],
        edges: Sequence[tuple[int, int]] = (),
        h: Sequence[float] = (),
        alpha1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.vessels = vessels
        self.edges = edges
        self.h = h
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the adaptation of every conductance to the pressure drop.

        The pressure drop is that of the current flows.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        rate = CallSpec(self._adaptation, (s.vessels.C, s.alpha1, s.alpha2, s.gamma), vectorized=True)
        return [RateLaw(s.vessels.C, rate)]

    def _adaptation(
        self,
        t: float,
        C: npt.NDArray[np.float64],
        alpha1: npt.NDArray[np.float64],
        alpha2: npt.NDArray[np.float64],
        gamma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the rates of the conductances at the pressures of the mass balance."""
        drops = self._pressure_drops(np.asarray(C, dtype=float))
        return C * (alpha1 * np.abs(drops) ** gamma - alpha2)

    def _pressure_drops(self, C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the pressure drop `p_i - p_j` along every edge.

        It solves `h = K p` with the first pressure at zero.
        """
        ends = np.asarray(self.edges, dtype=int).reshape(-1, 2)
        size = len(self.h)
        laplacian = np.zeros((size, size))
        np.add.at(laplacian, (ends[:, 0], ends[:, 0]), C)
        np.add.at(laplacian, (ends[:, 1], ends[:, 1]), C)
        np.add.at(laplacian, (ends[:, 0], ends[:, 1]), -C)
        np.add.at(laplacian, (ends[:, 1], ends[:, 0]), -C)
        pressures = np.zeros(size)
        pressures[1:] = np.linalg.lstsq(laplacian[1:, 1:], np.asarray(self.h, dtype=float)[1:], rcond=None)[0]
        return pressures[ends[:, 0]] - pressures[ends[:, 1]]


class AdaptiveFlowNetworkSystem(BaseSystem):
    """A square grid of vessels whose conductances adapt.

    It is fed at one corner and drained at every other node.

    The pressures balance the flows at every node, `h_i = sum_j C_ij (p_i - p_j)`, and every conductance grows with the
    dissipated power and decays, `dC_ij/dt = C_ij (alpha1 |p_i - p_j|^gamma - alpha2)`. Under a steady supply the
    network settles where every surviving vessel carries the same pressure drop `(alpha2 / alpha1)^(1 / gamma)`, so the
    conductances grow with the flow each vessel must carry: the vessels near the source thicken into trunks and the
    distal ones stay thin. The system has the conductances `vessels.C` on the edges of the grid.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import AdaptiveFlowNetworkSystem
        >>> AdaptiveFlowNetworkSystem(3).prepare().system_.elements[0].prepare().size_
        12
    """

    field = 'biology/physiology'
    topics = ('adaptive network', 'vasculature', 'kirchhoff', 'remodelling', 'ode')
    references = ('Hu and Cai 2013', 'Martens and Klemm 2017', 'Berner et al. 2023')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        side: int = 6,
        *,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        gamma: float = 2.0,
        inflow: float | None = None,
        seed: int | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.side = side
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma
        self.inflow = inflow
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of a square grid of vessels.

        It is fed at one corner and drained at every other node.
        """
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self.side, self.side))
        space = NetworkSpace(graph).prepare()
        sources, targets = space.edges_[None]
        n_nodes, n_edges = graph.number_of_nodes(), len(sources)
        supply = float(n_edges) if self.inflow is None else self.inflow
        h = np.full(n_nodes, -supply / (n_nodes - 1))
        h[0] = supply
        rng = np.random.default_rng(self.seed)
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
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            gamma=self.gamma,
        )
        return System(self.label, elements=[vessels], interactions=[adaptation])
