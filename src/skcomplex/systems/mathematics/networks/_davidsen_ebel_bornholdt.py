"""Davidsen-Ebel-Bornholdt network.

Triadic closure and node turnover at a fixed size give high clustering.
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
from ....core.components.interactions import Interaction
from ....core.components.laws import TopologyRequestSpec, UpdateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class ClosingNodeElement(Element):
    """Node of the Davidsen-Ebel-Bornholdt network, with its degree.

    A map of the system reads the degree from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import ClosingNodeElement
        >>> ClosingNodeElement('nodes').degree
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


class DavidsenEbelBornholdtSystem(BaseSystem):
    """The Davidsen-Ebel-Bornholdt network.

    Its triadic closure and node turnover give high clustering.

    The network starts as an Erdős-Rényi graph of `n_nodes` nodes and `links` links, `n_nodes` by default. Every step a
    node drawn uniformly joins two of its neighbours that are not yet joined, or, without such a pair, joins a node
    drawn uniformly. Then, with probability `turnover`, a node drawn uniformly leaves with its links and a new node
    takes its place with one link to a node drawn uniformly (Thurner et al. 2018, section 4.5.4.6). A closure always
    adds one link, and a turnover removes a node of mean degree `k` and adds one link, so the steady state has the mean
    degree `k = 1 + 1 / turnover`. Closing triangles keeps the clustering high, and since a node lies near many others
    in proportion to its degree, closure attaches preferentially, so hubs emerge as the turnover falls. The system has
    the groups `nodes` and `links`, the closure `closure`, and the turnover `turnover`.

    Args:
        n_nodes:
            The number of nodes.

        turnover:
            The probability that a node is replaced at a step.

        links:
            The number of links of the initial Erdős-Rényi graph, or `None` for `n_nodes`.

        random_state:
            The seed or generator of the initial graph.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import DavidsenEbelBornholdtSystem
        >>> trajectory = DavidsenEbelBornholdtSystem(50).prepare().simulate(until=20, random_state=0)
        >>> len(trajectory.get_network(-1).nodes)
        50
    """

    field = 'mathematics/networks'
    topics = ('network', 'triadic closure', 'clustering', 'node turnover', 'non-growing network')
    references = ('Davidsen, Ebel, and Bornholdt 2002', 'Thurner et al. 2018')
    until = 10000

    def __init__(
        self,
        n_nodes: int = 200,
        *,
        turnover: float = 0.1,
        links: int | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_nodes = n_nodes
        self.turnover = turnover
        self.links = links
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the Davidsen-Ebel-Bornholdt network, whose closure and turnover give high clustering."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        graph = nx.gnm_random_graph(self.n_nodes, self.n_nodes if self.links is None else self.links, seed=rng)
        closure = Interaction(
            'closure',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(TopologyRequestSpec('close-triangle', one=True), 1)],
        )
        replacement = Interaction(
            'turnover',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(TopologyRequestSpec('replace-node', one=True), self.turnover)],
        )
        space = NetworkSpace(graph)
        nodes = ClosingNodeElement('nodes', space=space)
        edges = Element('links', space=space, on='edges')
        degree = Interaction(
            'degree',
            roles={'target': 'nodes'},
            laws=lambda s: [UpdateLaw(s.target.degree, neighbors(1))],
        )
        self.system_ = System(
            self.label,
            elements=[nodes, edges],
            interactions=[closure, replacement, degree],
            space=space,
        )
