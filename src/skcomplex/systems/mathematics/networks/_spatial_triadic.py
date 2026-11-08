"""Spatial triadic percolation: nearby signed regulators make the giant component of a network on a torus move."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import SR, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import HyperedgeInteraction, Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, components, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec

CHUNK = 500
ACTIVITY_VISUAL = VisualSpec('sites', {'states': ['active'], 'low': 0.0, 'high': 1.0})


class PercolatingNodeElement(Element):
    """Node of a network with spatial triadic regulation, active when it belongs to the giant component.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the node is in the giant component of the active links, and 0 otherwise.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import PercolatingNodeElement
        >>> PercolatingNodeElement('nodes').active
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


class SignedLinkElement(Element):
    """Link regulated by signed nodes, with its activity and the numbers of its negative and positive regulators.

    Args:
        label:
            The label of the link group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        active:
            1 when the link is active, and 0 otherwise.

        negatives:
            The number of regulating nodes that let the link be active only while they are inactive, the first
            regulators of its hyperedge.

        positives:
            The number of regulating nodes that let the link be active while they are active, the regulators of its
            hyperedge after the negative ones.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SignedLinkElement
        >>> SignedLinkElement('links', negatives=2).negatives
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        negatives: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0,
        positives: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active
        self.negatives = negatives
        self.positives = positives


class SignedRegulationInteraction(HyperedgeInteraction):
    """Regulation of every link by signed regulating nodes, which let it be active with probability `p` in its phase.

    Every hyperedge `(i, j, r_1, ..., r_m)` holds the two ends of the link `(i, j)` and its regulating nodes, first its
    `negatives` negative regulators and then its `positives` positive ones, which the link reads. Further members pad
    the hyperedges to one size and regulate nothing, so links with several numbers of regulators share one size and one
    law writes every link once. In every step the link becomes active with probability `p` when none of its negative
    regulators is active and at least one of its positive regulators is active, or, for a link without positive
    regulators, when none of its negative regulators is active, and inactive otherwise.

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
        >>> from skcomplex.systems.mathematics.networks import SignedRegulationInteraction
        >>> SignedRegulationInteraction('regulation', nodes='nodes', links='links').p
        0.5
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
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.p = p
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the activity of the link from its regulating nodes, the members of the hyperedge after its ends."""
        negatives, positives = s.edge.negatives, s.edge.positives
        blocked, silent = SR(1), SR(1)
        for position, regulator in enumerate(s.others[1:]):
            blocked *= 1 - where(negatives > position, regulator.active, 0)
            positive = where(negatives > position, 0, where(negatives + positives > position, 1, 0))
            silent *= 1 - positive * regulator.active
        allowed = blocked * where(positives > 0, 1 - silent, 1)
        return [UpdateLaw(s.edge.active, bernoulli('retained', s.p) * allowed, phase=self.phase)]


class PercolationInteraction(Interaction):
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
        >>> from skcomplex.systems.mathematics.networks import PercolationInteraction
        >>> PercolationInteraction('percolation', nodes='nodes', links='links').links
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


class SpatialTriadicPercolationSystem(BaseSystem):
    """Spatial triadic percolation, where nearby signed regulators make the giant component form moving patterns.

    The nodes lie uniformly on a square torus of side `side` with `density` nodes per unit area, and two nodes at the
    distance `d` are linked with probability `c exp(-d / structural_range)`, a Waxman network, Eq. 1 of Millán et al.
    A node regulates a link positively with probability `positive exp(-d / regulatory_range)` and negatively with
    probability `negative exp(-d / regulatory_range)`, never both, from its distance `d` to the midpoint of the link,
    Eq. 2. Every step of the paper runs in two phases. In the phase `regulation` a link is active with probability `p`
    when none of its negative regulators is active and one of its positive regulators is, or, for the rare link
    without positive regulators, when none of its negative regulators is active. In the phase `percolation` a node is
    active when it belongs to the giant component of the active links. All nodes and links start active.

    The regulation is local, so the giant component forms clusters, stripes, and octopus-like patterns that move, and
    its size keeps changing instead of settling. At `p = 0.8` with `positive = negative` the patterns blink: a pattern
    deactivates the links it regulates negatively and the next step activates the region around it, so the active nodes
    of consecutive steps are anti-correlated. For long ranges the model becomes the triadic percolation of Erdős-Rényi
    networks with many regulators per link. The system has the activities `nodes.active` and `links.active`, and
    `links.negatives` and `links.positives`, the numbers of negative and positive regulators of every link, on a network
    whose nodes hold their positions.

    Args:
        side:
            The side of the square torus.

        density:
            The number of nodes per unit area.

        c:
            The prefactor of the probability of a structural link.

        positive:
            The prefactor of the probability of a positive regulation.

        negative:
            The prefactor of the probability of a negative regulation.

        structural_range:
            The typical length of the structural links, `d0`.

        regulatory_range:
            The typical length of the regulations, `dr`.

        p:
            The probability that a link the regulation allows is active.

        random_state:
            The seed or generator of the positions, the links, and the regulators.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.mathematics.networks import SpatialTriadicPercolationSystem
        >>> system = SpatialTriadicPercolationSystem(1.0).prepare().system_
        >>> trajectory = simulate(system, until=2, random_state=0)
        >>> giant = trajectory.values('nodes.active').mean(axis=1)
        >>> bool(giant[0] == 1.0 and giant[1] < 0.1)
        True
    """

    field = 'mathematics/networks'
    topics = ('network', 'hypergraph', 'percolation', 'triadic interactions', 'spatial network', 'pattern formation')
    references = ('Millán, Sun, Torres, and Bianconi 2023', 'Waxman 1988')
    until = 100
    dt = 1
    observables = (ObservableSpec('giant', group='nodes', variable='active'),)

    def __init__(
        self,
        side: float = 4.0,
        *,
        density: float = 100.0,
        c: float = 0.4,
        positive: float = 0.2,
        negative: float = 0.2,
        structural_range: float = 0.25,
        regulatory_range: float = 0.25,
        p: float = 0.8,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.side = side
        self.density = density
        self.c = c
        self.positive = positive
        self.negative = negative
        self.structural_range = structural_range
        self.regulatory_range = regulatory_range
        self.p = p
        self.random_state = random_state

    @staticmethod
    def _torus_distances(first: np.ndarray, second: np.ndarray, side: float) -> np.ndarray:
        """Return the distances on a square torus between every point of a first and of a second array of points."""
        offsets = np.abs(first[:, None, :] - second[None, :, :])
        offsets = np.minimum(offsets, side - offsets)
        return np.hypot(offsets[..., 0], offsets[..., 1])

    @staticmethod
    def _waxman_torus(n_nodes: int, side: float, c: float, scale: float, rng: np.random.Generator) -> nx.Graph:
        """Return nodes placed uniformly on a square torus, each pair linked with probability `c exp(-d / scale)`."""
        positions = rng.uniform(0.0, side, size=(n_nodes, 2))
        graph = nx.Graph()
        graph.add_nodes_from((node, {'pos': tuple(positions[node])}) for node in range(n_nodes))
        for start in range(0, n_nodes, CHUNK):
            rows = np.arange(start, min(start + CHUNK, n_nodes))
            distances = SpatialTriadicPercolationSystem._torus_distances(positions[rows], positions, side)
            linked = (rng.uniform(size=distances.shape) < c * np.exp(-distances / scale)) & (
                np.arange(n_nodes)[None, :] > rows[:, None]
            )
            graph.add_edges_from((int(rows[row]), int(other)) for row, other in zip(*np.nonzero(linked), strict=True))
        return graph

    @staticmethod
    def _spatial_regulators(
        graph: nx.Graph,
        side: float,
        chances: tuple[float, float],
        scale: float,
        rng: np.random.Generator,
    ) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
        """Return a hyperedge of every link with its negative and then its positive regulators, padded to one size.

        A node regulates a link positively with probability `c+ exp(-d / scale)` and negatively with probability `c-
        exp(-d / scale)`, never both, from its distance `d` to the midpoint of the link on the torus. Every hyperedge is
        padded with the first end of its link up to the size of the largest one.
        """
        positive, negative = chances
        positions = np.asarray([graph.nodes[node]['pos'] for node in graph.nodes], dtype=float)
        ends = np.asarray(list(graph.edges), dtype=np.intp).reshape(-1, 2)
        steps = positions[ends[:, 1]] - positions[ends[:, 0]]
        midpoints = (positions[ends[:, 0]] + (steps - side * np.round(steps / side)) / 2) % side
        regulators: list[tuple[int, ...]] = []
        negatives, positives = [], []
        for start in range(0, len(ends), CHUNK):
            decay = np.exp(
                -SpatialTriadicPercolationSystem._torus_distances(midpoints[start : start + CHUNK], positions, side)
                / scale,
            )
            draws = rng.uniform(size=decay.shape)
            for row, (chance, draw) in enumerate(zip(decay, draws, strict=True)):
                against = np.flatnonzero((draw >= positive * chance) & (draw < (positive + negative) * chance))
                towards = np.flatnonzero(draw < positive * chance)
                regulators.append((*ends[start + row].tolist(), *against.tolist(), *towards.tolist()))
                negatives.append(against.size)
                positives.append(towards.size)
        width = max((len(item) for item in regulators), default=2)
        hyperedges = [item + (item[0],) * (width - len(item)) for item in regulators]
        return hyperedges, np.asarray(negatives), np.asarray(positives)

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        graph = self._waxman_torus(round(self.density * self.side**2), self.side, self.c, self.structural_range, rng)
        hyperedges, negatives, positives = self._spatial_regulators(
            graph,
            self.side,
            (self.positive, self.negative),
            self.regulatory_range,
            rng,
        )
        space = NetworkSpace(graph, hyperedges=hyperedges)
        nodes = LatticeElement('nodes', element=PercolatingNodeElement, space=space)
        values = {'negatives': negatives, 'positives': positives}
        links = GroupElement('links', element=SignedLinkElement, space=space, on='edges', values=values)
        interactions = [
            SignedRegulationInteraction('regulation', nodes='nodes', links='links', p=self.p, phase='regulation'),
            PercolationInteraction('percolation', nodes='nodes', links='links', phase='percolation'),
        ]
        time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('regulation'), PhaseSpec('percolation'))))
        return System(self.label, elements=[nodes, links], interactions=interactions, time=time)
