"""Triadic percolation.

Regulating nodes switch the links of a network on and off from the giant component.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import HyperedgeInteraction, Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, components, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ..common.networks import RegulatedNodeElement


class RegulatedLinkElement(Element):
    """Link of a network with triadic regulation.

    The link has its activity and the sign of its regulating node.

    Args:
        label:
            The label of the link group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the link is active, and 0 otherwise.

        sign:
            1 when the regulating node lets the link be active while the node is active, and -1 when it lets it be
            active while the node is inactive.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import RegulatedLinkElement
        >>> RegulatedLinkElement('links', sign=-1).sign
        -1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        sign: Annotated[float, ParameterSpec(domain=FiniteEnumeratedSet([-1, 1]))] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active
        self.sign = sign


class GiantComponentInteraction(Interaction):
    """Activity of every node in its phase.

    The activity is 1 in the giant component of the active links and 0 outside it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes, whose activity the interaction sets.

        links:
            The path of the group on the edges, whose activity selects the links of the components.

        phase:
            The phase of the step the interaction runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import GiantComponentInteraction
        >>> GiantComponentInteraction('percolation', nodes='nodes', links='links').links
        'links'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the membership of the giant component of the active links."""
        return [UpdateLaw(s.nodes.active, components(s.links.active, reduce='giant'), phase=self.phase)]


class TriadicRegulationInteraction(HyperedgeInteraction):
    """Regulation of every link by a third node.

    The node lets the link be active with probability `p` in its phase.

    Every hyperedge `(i, j, r)` holds the two ends of the link `(i, j)` and its regulating node `r`. In every step the
    link becomes active with probability `p` when a positive regulator is active or a negative regulator is inactive,
    and inactive otherwise.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes, whose activity the regulators read.

        links:
            The path of the group on the edges, whose activity the regulation sets.

        p:
            The probability that a link the regulation allows is active.

        phase:
            The phase of the step the regulation runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import TriadicRegulationInteraction
        >>> TriadicRegulationInteraction('regulation', nodes='nodes', links='links').size
        3
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        p: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.links = links
        self.p = p
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the activity of the link from its regulating node.

        The regulating node is the third member of the hyperedge.
        """
        regulator = s.others[1].active
        allowed = where(s.edge.sign > 0, regulator, 1 - regulator)
        return [UpdateLaw(s.edge.active, bernoulli('retained', s.p) * allowed, phase=self.phase)]


class TriadicPercolationSystem(BaseSystem):
    """Triadic percolation.

    Regulating nodes switch links on and off from the giant component.

    Every link of an Erdős-Rényi graph has one regulating node, drawn at random among the other nodes, which is
    negative with probability `negative` and positive otherwise. The link and its regulator form a hyperedge of size
    three. Every step is a time step of Millán et al. in two phases. In the phase `regulation` every link is active
    with probability `p` when its positive regulator is active or its negative regulator is inactive, and inactive
    otherwise. In the phase `percolation` a node is active when it belongs to the giant component of the links the
    regulation of the same step left active. All nodes and links start active.

    With the fraction `R` of nodes in the giant component, the links are active with probability `p_L = p ((1 -
    negative) R + negative (1 - R))`, and `R` follows the map `R' = S(c p_L)`, where `S(m)` solves `S = 1 - exp(-m S)`
    for the mean degree `c`. With negative regulators alone the map decreases, so above the percolation point `p = 1 /
    c` its fixed point is unstable and `R` alternates between a large and a small giant component, a cycle of period
    two. Below it the giant component vanishes. The system has the activities `nodes.active` and `links.active` and the
    signs `links.sign`.

    Args:
        n_nodes:
            The number of nodes.

        mean_degree:
            The mean degree of the Erdős-Rényi graph.

        p:
            The probability that a link the regulation allows is active.

        negative:
            The probability that the regulator of a link is negative.

        random_state:
            The seed or generator of the graph, the regulators, and their signs.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import TriadicPercolationSystem
        >>> system = TriadicPercolationSystem(200, p=0.8).prepare().system_
        >>> trajectory = simulate(system, until=3, random_state=0)
        >>> giant = trajectory.get_values('nodes.active').mean(axis=1)
        >>> bool(giant[1] < 0.5 < giant[2])
        True
    """

    field = 'mathematics/networks'
    topics = ('network', 'hypergraph', 'percolation', 'triadic interactions', 'period doubling')
    references = ('Sun et al. 2023', 'Millán, Sun, Torres, and Bianconi 2023')
    until = 30
    dt = 1
    observables = (ObservableSpec('giant', group='nodes', variable='active'),)

    def __init__(
        self,
        n_nodes: int = 1000,
        *,
        mean_degree: float = 4.0,
        p: float = 0.5,
        negative: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.mean_degree = mean_degree
        self.p = p
        self.negative = negative
        self.random_state = random_state

    @staticmethod
    def _get_regulated_links(
        graph: nx.Graph,
        negative: float,
        rng: np.random.Generator,
    ) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        """Return a hyperedge of every link with a regulator drawn among the other nodes.

        The sign of every regulator is returned too.
        """
        size = graph.number_of_nodes()
        hyperedges = []
        for first, second in graph.edges:
            regulator = int(rng.integers(size - 2))
            for end in sorted((first, second)):
                regulator += int(regulator >= end)
            hyperedges.append((first, second, regulator))
        signs = np.where(rng.uniform(size=len(hyperedges)) < negative, -1, 1)
        return hyperedges, signs

    def _build_system(self) -> CompositeSystem:
        """Return triadic percolation.

        Regulating nodes switch links on and off from the giant component.
        """
        rng = np.random.default_rng(self.random_state)
        graph = nx.gnp_random_graph(self.n_nodes, self.mean_degree / (self.n_nodes - 1), seed=rng)
        hyperedges, signs = self._get_regulated_links(graph, self.negative, rng)
        space = NetworkSpace(graph, hyperedges=hyperedges)
        nodes = GroupElement('nodes', element=RegulatedNodeElement, space=space, on='sites')
        links = GroupElement('links', element=RegulatedLinkElement, space=space, on='edges', values={'sign': signs})
        interactions = [
            TriadicRegulationInteraction('regulation', nodes='nodes', links='links', p=self.p, phase='regulation'),
            GiantComponentInteraction('percolation', nodes='nodes', links='links', phase='percolation'),
        ]
        time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('regulation'), PhaseSpec('percolation'))))
        return CompositeSystem(self.label, elements=[nodes, links], interactions=interactions, time=time)
