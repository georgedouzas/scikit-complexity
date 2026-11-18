"""Duplication-deletion network of Jensen.

Nodes die and duplicate, and the connectance settles at a steady value.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class DuplicatingNodeElement(Element):
    """Node of the duplication-deletion network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DuplicatingNodeElement
        >>> DuplicatingNodeElement('nodes').degree
        0.0
    """

    visual = DEGREE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        degree: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.degree = degree


class DuplicationDeletionSystem(BaseSystem):
    """The steady-state duplication-deletion network of Jensen.

    Its connectance settles at Eq. 8.84.

    The network starts as an Erdős-Rényi graph of `n_nodes` nodes and the link probability `density`. Every step a node
    drawn uniformly leaves with its links with probability `kill`, and a parent drawn uniformly among the nodes left is
    duplicated: the offspring inherits each link of the parent with probability `inherit`, joins the parent with
    probability `parent`, and joins every node the parent does not join with probability `new` (Jensen 2023, section
    8.5.1). With `kill = 1` the network keeps `N` nodes, and its connectance converges to
    `C = (new (N - 2) + parent) / (N - 1 + (new - inherit) (N - 2))`, Eq. 8.84. Accurate duplication, with `inherit`
    near 1 and `new` near 0, turns the binomial degrees of noisy duplication into a cut-off power law. With `kill < 1`
    the network grows by `1 - kill` nodes per step on average.

    The system has the groups `nodes` and `links`, the deletion `death`, and the duplication `birth`.

    Args:
        n_nodes:
            The number of nodes.

        kill:
            The probability that the drawn node leaves at a step.

        inherit:
            The probability that the offspring inherits a link of its parent.

        parent:
            The probability that the offspring joins its parent.

        new:
            The probability that the offspring joins a node its parent does not join.

        density:
            The link probability of the initial Erdős-Rényi graph.

        random_state:
            The seed or generator of the initial graph.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DuplicationDeletionSystem
        >>> trajectory = DuplicationDeletionSystem(30).prepare().simulate(until=20, random_state=0)
        >>> len(trajectory.get_network(-1).nodes)
        30
    """

    field = 'mathematics/networks'
    topics = ('network', 'duplication', 'deletion', 'evolution', 'protein interaction network', 'connectance')
    references = ('Jensen 2023', 'Sole et al. 2002')
    until = 4000

    def __init__(
        self,
        n_nodes: int = 100,
        *,
        kill: float = 1.0,
        inherit: float = 0.5,
        parent: float = 0.5,
        new: float = 0.01,
        density: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.kill = kill
        self.inherit = inherit
        self.parent = parent
        self.new = new
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the steady-state duplication-deletion network of Jensen."""
        super()._prepare(context)
        graph = nx.gnp_random_graph(self.n_nodes, self.density, seed=np.random.default_rng(self.random_state))
        death = TopologyInteraction('death', target='nodes', action='remove-node', one=True, probability=self.kill)
        birth = TopologyInteraction(
            'birth',
            target='nodes',
            action='duplicate-node',
            one=True,
            inherit=self.inherit,
            parent=self.parent,
            independent=True,
            toward=lambda node: self.new,
        )
        space = NetworkSpace(graph)
        nodes = DuplicatingNodeElement('nodes', space=space)
        edges = Element('links', space=space, on='edges')
        degree = MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))
        self.system_ = System(
            self.label,
            elements=[nodes, edges],
            interactions=[death, birth, degree],
            space=space,
        )
