"""Intercellular exchange: substances passed between cells in a closed contour and on networks of junctions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import hill
from ....core.components.spaces import ContinuousTime, DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.components.types import Target
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class ContourCellElement(Element):
    """Logistic cell of a closed contour that takes up the signalling molecules of the next cell through a power map.

    `ContourUptakeInteraction` iterates `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^{p_i}`, where the logistic map
    is the intracellular dynamics, `c_i` the share of the exchange with the environment between the cells, and `p_i`
    the affinity of the cell for the uptake. The last cell of the contour takes up from the first.

    Args:
        label:
            The label of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration of signalling molecules in the cell, between 0 and 1.

        r:
            The logistic parameter, the disposition of the environment to communication.

        c:
            The coupling of the cell to the next one.

        p:
            The affinity of the cell for the uptake of signalling molecules.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ContourCellElement
        >>> ContourCellElement('cells').r
        2.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.5,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
        self.c = c
        self.p = p


class ExchangingCellElement(Element):
    """Cell that metabolizes a substance and exchanges it with its neighbours through gap junctions and nanotubes.

    The concentration follows `dx_i/dt = -alpha_i x_i / (beta_i + x_i) + Σ_j [g (x_j - x_i) + xi A_ij x_j]`, where the
    first term is Michaelis-Menten metabolism, the one law of the cell, `g` the exchange coefficient of the gap
    junctions, and `xi` the strength of the tunnelling nanotubes along the same links. The sum over the neighbours is a
    `CouplingInteraction` of a network of cells.

    Args:
        label:
            The label of the field of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration of the substance.

        alpha:
            The maximal rate of metabolism.

        beta:
            The half-saturation concentration of metabolism.

        g:
            The exchange coefficient of the gap junctions.

        xi:
            The strength of the nanotubes.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ExchangingCellElement
        >>> ExchangingCellElement('cells').g
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        xi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.g = g
        self.xi = xi

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the metabolism of the substance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, -s.alpha * hill(s.x, s.beta))]


class ContourUptakeInteraction(CouplingInteraction):
    """Uptake of every cell of a contour from the next cell, `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^p_i`.

    It is the map form of coupling to the next member of the group, with the power `x_{i+1}^{p_i}` of the next cell
    combined with the logistic map of the cell. The last cell of the contour takes up from the first.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of cells.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.physiology import ContourCellElement, ContourUptakeInteraction
        >>> from skcomplex.core.engine.runtime import simulate
        >>> cells = GroupElement('cells', element=ContourCellElement, size=2, values={'x': [0.25, 1.0], 'c': 0.5})
        >>> uptake = ContourUptakeInteraction('uptake', target='cells')
        >>> system = System(elements=[cells], interactions=[uptake], time=DiscreteTime())
        >>> simulate(system, until=1).values('cells.x')[-1].round(4).tolist()
        [0.7344, 0.25]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            read='x',
            write='x',
            function=self._get_power,
            mode='next',
            form='map',
            combine=self._get_mix,
        )

    @staticmethod
    def _get_power(namespace: SimpleNamespace) -> Target:
        """Return the power uptake of the next cell, with the affinity of the cell."""
        return namespace.other.x**namespace.member.p

    @staticmethod
    def _get_mix(namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the logistic map of the cell mixed with the uptake from the next cell."""
        member = namespace.member
        return (1 - member.c) * member.r * member.x * (1 - member.x) + member.c * reduction


def _exchange(s: SimpleNamespace) -> Expression:
    """Return the exchange of a cell with a neighbour, through their gap junction and the nanotube between them."""
    exchange: Expression = s.member.g * (s.other.x - s.member.x) + s.member.xi * s.other.x
    return exchange


class ClosedContourCellExchangeSystem(BaseSystem):
    """The exchange of signalling molecules between logistic cells arranged in a closed contour.

    Every cell runs the logistic map on its concentration and takes up a share `c_i` of its next state from the next
    cell through the power map `x_{i+1}^{p_i}`, `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^{p_i}`. Mihailović et
    al. show that an equilibrium is asymptotically stable for every coupling when every `x_i` lies in `((r - 1) / 2r, (r
    + 1) / 2r)` and above `p_{i-1}^(1 / (1 - p_{i-1}))`. In this domain the cells synchronize to a common concentration,
    and a chaotic environment, `r` near 4, desynchronizes them. The system has the concentrations `cells.x`, cell `i +
    1` next to cell `i`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ClosedContourCellExchangeSystem
        >>> ClosedContourCellExchangeSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'biology/physiology'
    topics = ('map', 'logistic', 'coupled maps', 'ring', 'synchronization', 'cell communication')
    references = ('Mihailović et al. 2023',)
    until = 200
    dt = 1

    def __init__(
        self,
        n_cells: int = 8,
        *,
        r: float = 2.5,
        c: float | Sequence[float] = 0.1,
        p: float | Sequence[float] = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_cells = n_cells
        self.r = r
        self.c = c
        self.p = p
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system."""
        rng = np.random.default_rng(self.random_state)
        cells = GroupElement(
            'cells',
            element=ContourCellElement,
            size=self.n_cells,
            values={
                'x': rng.uniform(0.2, 0.8, self.n_cells),
                'r': self.r,
                'c': np.broadcast_to(np.asarray(self.c, dtype=float), self.n_cells).copy(),
                'p': np.broadcast_to(np.asarray(self.p, dtype=float), self.n_cells).copy(),
            },
        )
        uptake = ContourUptakeInteraction('uptake', target='cells')
        return System(self.label, elements=[cells], interactions=[uptake], time=DiscreteTime())


class NanotubeGapJunctionExchangeSystem(BaseSystem):
    """The intercellular exchange of a substance through gap junctions and tunnelling nanotubes on a network.

    Every cell metabolizes the substance at the Michaelis-Menten rate `alpha x / (beta + x)` and exchanges it with its
    neighbours, by diffusion `g (x_j - x_i)` through gap junctions and by the transfer `xi x_j` through nanotubes. The
    state without substance is stable while the largest eigenvalue of `xi A - g L` stays below `alpha / beta`, where `A`
    is the adjacency and `L` the Laplacian of the network. Past the threshold `xi_crit` the nanotubes destabilize the
    communication and the concentrations grow. On a regular graph of degree `k`, `xi_crit = alpha / (beta k)`. The
    system has the concentrations `cells.x`, one value per node.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.physiology import NanotubeGapJunctionExchangeSystem
        >>> NanotubeGapJunctionExchangeSystem(nx.cycle_graph(6), xi=0.01).prepare().system_.elements[0].prepare().size_
        6
    """

    field = 'biology/physiology'
    topics = ('network', 'cell communication', 'gap junctions', 'nanotubes', 'stability', 'ode')
    references = ('Mihailović et al. 2023',)
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        alpha: float = 1.0,
        beta: float = 10.0,
        g: float = 0.1,
        xi: float = 0.0,
        x: float | Sequence[float] | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.g = g
        self.xi = xi
        self.x = x
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system."""
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        if graph is None:
            graph = nx.random_partition_graph([10] * 10, 0.297, 0.0033, seed=int(rng.integers(2**31)))
        size = graph.number_of_nodes()
        initial = (
            rng.uniform(0.0, 1.0, size)
            if self.x is None
            else np.broadcast_to(np.asarray(self.x, dtype=float), size).copy()
        )
        values = {
            'x': initial,
            'alpha': self.alpha,
            'beta': self.beta,
            'g': self.g,
            'xi': self.xi,
        }
        cells = FieldElement('cells', element=ExchangingCellElement, space=NetworkSpace(graph), values=values)
        exchange = CouplingInteraction('exchange', target='cells', read='x', write='x', function=_exchange)
        return System(self.label, elements=[cells], interactions=[exchange], time=ContinuousTime())
