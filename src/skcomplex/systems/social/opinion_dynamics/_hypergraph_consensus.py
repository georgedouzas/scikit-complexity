"""Nonlinear consensus on temporal hypergraphs.

Opinions are pulled by triangles that switch on and off in time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import HyperedgeInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class OpinionElement(Element):
    """Node of a consensus process with a continuous opinion.

    Args:
        label:
            The label of the node or the group.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        x:
            The initial opinion.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import OpinionElement
        >>> OpinionElement('node', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class ScheduleElement(Element):
    """Triangle of a temporal hypergraph.

    It is active in every other half of a period from its own phase.

    Args:
        label:
            The label of the triangle or the group.

        visual:
            How the triangle is drawn, or `None` for `default_visual_`.

        phase:
            The phase of the schedule, in radians.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ScheduleElement
        >>> ScheduleElement('triangle', phase=1.0).phase
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phase: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phase = phase


TRIANGLE = 3


class ThreeBodyConsensusInteraction(HyperedgeInteraction):
    """Nonlinear three-body consensus of Neuhäuser, Mellor, and Lambiotte.

    It acts on the triangles active at the time. The focal node `i` of an active triangle `ijk` moves towards the two
    others at the rate `s(|x_j - x_k|) (x_j + x_k - 2 x_i)`, with `s(d) = exp(reinforcement * d)`, so a pair that agrees
    pulls harder for a negative reinforcement. A triangle is active when `sin(2 pi t / period + phase) > 0`, with the
    phase of the triangle member of the group on hyperedges.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of nodes.

        triangles:
            The path of the group of the schedules of the triangles.

        reinforcement:
            The exponent of the modulating function `s`.

        period:
            The period of the schedules.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ThreeBodyConsensusInteraction
        >>> ThreeBodyConsensusInteraction('consensus', target='nodes', triangles='triangles').size
        3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        triangles: Annotated[str, RoleSpec()],
        reinforcement: Annotated[float, ParameterSpec()] = 0.0,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, size=TRIANGLE)
        self.target = target
        self.triangles = triangles
        self.reinforcement = reinforcement
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the pull of the two other members of an active triangle.

        The pull acts on the focal node.
        """
        x = s.member.x
        first, second = (other.x for other in s.others)
        active = where(sin(2 * pi * s.t / s.period + s.hyperedge.phase) > 0, 1, 0)
        pull = exp(s.reinforcement * abs(first - second)) * (first + second - 2 * x)
        return [RateLaw(x, active * pull)]


def _spread(s: SimpleNamespace) -> object:
    """Return the squared distance of an opinion from zero.

    The variance of the opinions averages it.
    """
    return s.x**2


class HypergraphConsensusSystem(BaseSystem):
    """The nonlinear three-body consensus system on a random hypergraph of triangles.

    The triangles switch on and off. Every triple of nodes is a triangle with probability `triangles`, and every
    triangle is active in alternate half periods from a random phase. On an active triangle `ijk`, node `i` follows
    `x_i' = s(|x_j - x_k|) (x_j + x_k - 2 x_i)` with `s(d) = exp(reinforcement * d)`. Without reinforcement the pulls of
    a triangle on its three members sum to zero, so the mean opinion stays fixed whatever the schedules. With
    reinforcement the pull depends on the member it acts on, and the mean opinion drifts towards the pairs that agree,
    for a negative reinforcement, while the opinions still reach a consensus on a connected hypergraph.

    The system has the opinions `nodes.x`, the mean opinion `mean_opinion`, and the mean squared opinion `spread`.

    Args:
        n_nodes:
            The number of nodes.

        triangles:
            The probability that a triple of nodes is a triangle.

        reinforcement:
            The exponent of the modulating function `s`.

        period:
            The period of the schedules of the triangles.

        seed:
            The seed of the hypergraph, the phases, and the opinions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import HypergraphConsensusSystem
        >>> system = HypergraphConsensusSystem(8, triangles=0.5).prepare().system_
        >>> system.elements[0].prepare().size_
        8
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'consensus', 'hypergraph', 'temporal network')
    references = ('Neuhäuser, Mellor, and Lambiotte 2020', 'Majhi et al. 2022')
    until = 10.0
    dt = 0.1
    observables = (
        ObservableSpec('mean_opinion', group='nodes', variable='x'),
        ObservableSpec('spread', _spread, group='nodes'),
    )

    def __init__(
        self,
        n_nodes: int = 30,
        *,
        triangles: float = 0.05,
        reinforcement: float = -1.0,
        period: float = 2.0,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.triangles = triangles
        self.reinforcement = reinforcement
        self.period = period
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.seed)
        triads = [
            triad for triad in itertools.combinations(range(self.n_nodes), TRIANGLE) if rng.random() < self.triangles
        ]
        space = NetworkSpace(nx.complete_graph(self.n_nodes), hyperedges=triads)
        nodes = LatticeElement(
            'nodes',
            element=OpinionElement,
            space=space,
            values={'x': rng.uniform(-1.0, 1.0, self.n_nodes)},
            visual=VisualSpec('sites', {'states': ['x'], 'low': -1.0, 'high': 1.0}),
        )
        schedules = GroupElement(
            'triangles',
            element=ScheduleElement,
            space=space,
            on='hyperedges',
            values={'phase': rng.uniform(0.0, 2 * math.pi, len(triads))},
        )
        consensus = ThreeBodyConsensusInteraction(
            'consensus',
            target='nodes',
            triangles='triangles',
            reinforcement=self.reinforcement,
            period=self.period,
        )
        return System(self.label, elements=[nodes, schedules], interactions=[consensus])
