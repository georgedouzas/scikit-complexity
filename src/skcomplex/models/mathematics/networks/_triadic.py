"""Triadic percolation: regulating nodes switch the links of a network on and off from the giant component."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.elements import GroupElement, LatticeElement
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._giant_component import GiantComponentInteraction
from ._regulated_link import RegulatedLinkElement
from ._regulated_node import RegulatedNodeElement
from ._triadic_regulation import TriadicRegulationInteraction


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
