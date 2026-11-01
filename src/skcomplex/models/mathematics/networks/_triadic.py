"""Triadic percolation: regulating nodes switch the links of a network on and off from the giant component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement, LatticeElement
from ....core.interactions import HyperedgeInteraction, Interaction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import bernoulli, components, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

ACTIVITY_VISUAL = VisualSpec('sites', {'states': ['active'], 'low': 0.0, 'high': 1.0})


class RegulatedNodeElement(Element):
    """Node of a network with triadic regulation, active when it belongs to the giant component of the active links.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the node is in the giant component of the active links, and 0 otherwise.

    Examples:
        >>> from skcomplex.models.mathematics.networks import RegulatedNodeElement
        >>> RegulatedNodeElement('nodes').active
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = ACTIVITY_VISUAL,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active


class RegulatedLinkElement(Element):
    """Link of a network with triadic regulation, with its activity and the sign of its regulating node.

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
        >>> from skcomplex.models.mathematics.networks import RegulatedLinkElement
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


class TriadicRegulationInteraction(HyperedgeInteraction):
    """Regulation of every link by a third node, which lets the link be active with probability `p` in its phase.

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
        >>> from skcomplex.models.mathematics.networks import TriadicRegulationInteraction
        >>> TriadicRegulationInteraction('regulation', nodes='nodes', links='links').check().size
        3
    """

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
        """Return the activity of the link from its regulating node, the third member of the hyperedge."""
        regulator = s.others[1].active
        allowed = where(s.edge.sign > 0, regulator, 1 - regulator)
        return [UpdateLaw(s.edge.active, bernoulli('retained', s.p) * allowed, phase=self.phase)]


class GiantComponentInteraction(Interaction):
    """Activity of every node in its phase, 1 in the giant component of the active links and 0 outside it.

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
        >>> from skcomplex.models.mathematics.networks import GiantComponentInteraction
        >>> GiantComponentInteraction('percolation', nodes='nodes', links='links').links
        'links'
    """

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


def _giant_observables(system: System) -> list[ObservableSpec]:
    """Return the fraction of nodes in the giant component."""
    del system
    return [ObservableSpec('giant', group='nodes', variable='active')]


def _regulated_links(
    graph: nx.Graph,
    negative: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Return a hyperedge of every link with a regulator drawn among the other nodes, and the sign of every one."""
    size = graph.number_of_nodes()
    hyperedges = []
    for first, second in graph.edges:
        regulator = int(rng.integers(size - 2))
        for end in sorted((first, second)):
            regulator += int(regulator >= end)
        hyperedges.append((first, second, regulator))
    signs = np.where(rng.uniform(size=len(hyperedges)) < negative, -1, 1)
    return hyperedges, signs


@register_model(
    field='mathematics/networks',
    tags=('network', 'hypergraph', 'percolation', 'triadic interactions', 'period doubling'),
    references=('Sun et al. 2023', 'Millán, Sun, Torres, and Bianconi 2023'),
    defaults={'until': 30, 'dt': 1},
    observables=_giant_observables,
)
def triadic_percolation(
    n_nodes: int = 1000,
    *,
    mean_degree: float = 4.0,
    p: float = 0.5,
    negative: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return triadic percolation, where regulating nodes switch links on and off from the giant component.

    Every link of an Erdős-Rényi graph has one regulating node, drawn at random among the other nodes, which is
    negative with probability `negative` and positive otherwise. The link and its regulator form a hyperedge of size
    three. Every step is a time step of Millán et al. in two phases. In the phase `regulation` every link is active
    with probability `p` when its positive regulator is active or its negative regulator is inactive, and inactive
    otherwise. In the phase `percolation` a node is active when it belongs to the giant component of the links the
    regulation of the same step left active. All nodes and links start active.

    With the fraction `R` of nodes in the giant component, the links are active with probability
    `p_L = p ((1 - negative) R + negative (1 - R))`, and `R` follows the map `R' = S(c p_L)`, where `S(m)` solves
    `S = 1 - exp(-m S)` for the mean degree `c`. With negative regulators alone the map decreases, so above the
    percolation point `p = 1 / c` its fixed point is unstable and `R` alternates between a large and a small giant
    component, a cycle of period two. Below it the giant component vanishes.

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

    Returns:
        system:
            The model, with the activities `nodes.active` and `links.active` and the signs `links.sign`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import triadic_percolation
        >>> system = triadic_percolation(200, p=0.8)
        >>> trajectory = Simulation(system, random_state=0).check().prepare().run(until=3).trajectory_
        >>> giant = trajectory.values('nodes.active').mean(axis=1)
        >>> bool(giant[1] < 0.5 < giant[2])
        True
    """
    rng = np.random.default_rng(random_state)
    graph = nx.gnp_random_graph(n_nodes, mean_degree / (n_nodes - 1), seed=rng)
    hyperedges, signs = _regulated_links(graph, negative, rng)
    space = NetworkSpace(graph, hyperedges=hyperedges)
    nodes = LatticeElement('nodes', element=RegulatedNodeElement, space=space)
    links = GroupElement('links', element=RegulatedLinkElement, space=space, on='edges', values={'sign': signs})
    interactions = [
        TriadicRegulationInteraction('regulation', nodes='nodes', links='links', p=p, phase='regulation'),
        GiantComponentInteraction('percolation', nodes='nodes', links='links', phase='percolation'),
    ]
    time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('regulation'), PhaseSpec('percolation'))))
    return System(elements=[nodes, links], interactions=interactions, time=time)
