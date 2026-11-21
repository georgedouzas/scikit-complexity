"""Cascade systems: adoption that spreads when enough neighbours have adopted."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import NetworkSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ThresholdNodeView(AgentViewProtocol, Protocol):
    """View of a node of the Watts cascade system.

    Its states and parameters are attributes.
    """

    active: float
    threshold: float


class ThresholdActivationInteraction(Interaction):
    """Activation of the Watts cascade system.

    A node becomes active for good once the share of active nodes among its neighbouring nodes reaches its threshold.
    The nodes update one after the other in a new random order every step, each reading the activations the nodes before
    it set, so a cascade can cross several nodes in one step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of nodes.

    Examples:
        >>> from skcomplex.systems.social.spreading import ThresholdActivationInteraction
        >>> ThresholdActivationInteraction('activation', target='nodes').target
        'nodes'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the activation of the nodes, one after the other in a random order, as the rule of the system.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self._activate)]

    def _activate(self, system: SystemView) -> None:
        """Activate the nodes one after the other in a new random order."""
        nodes = cast(list[ThresholdNodeView], system.get_agents('target'))
        for index in system.rng.permutation(len(nodes)):
            node = nodes[int(index)]
            if node.alive:
                node.active = self._get_active(node)

    @staticmethod
    def _get_active(agent: ThresholdNodeView) -> float:
        """Return 1 for a node that is or becomes active.

        It returns 0 for one that stays inactive.
        """
        neighbors = cast(list[ThresholdNodeView], [item for item in agent.get_neighbors() if item.group == agent.group])
        if agent.active or not neighbors:
            return float(agent.active)
        return float(sum(neighbor.active for neighbor in neighbors) / len(neighbors) >= agent.threshold)


class ThresholdNodeElement(Element):
    """Node of the Watts cascade system.

    The activation turns it active once enough of its neighbours are active.

    Args:
        label:
            The label of the agent or group.

        active:
            Whether the node starts active, one per node or shared.

        threshold:
            The share of active neighbours at which the node becomes active, one per node or shared.

        size:
            The number of nodes, or `None` for one.

    Examples:
        >>> from skcomplex.systems.social.spreading import ThresholdNodeElement
        >>> group = ThresholdNodeElement('nodes', threshold=0.18, size=10)
        >>> group.label, group.threshold, group.size
        ('nodes', 0.18, 10)
    """

    visual = VisualSpec('agents', {'color_by': 'active', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        active: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        threshold: Annotated[object, RealSet.closed_open(0, oo)] = 0.18,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.active = active
        self.threshold = threshold


class WattsCascadesSystem(BaseSystem):
    """The Watts threshold system of global cascades started by a few random seeds.

    The system has the group `nodes`.

    Args:
        graph:
            The NetworkX graph, or `None` for a random graph of 200 nodes with mean degree 4.

        threshold:
            The share of active neighbours at which a node becomes active.

        seeds:
            The number of nodes active at the start.

        random_state:
            The seed or generator of the default graph and of the seeds.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.spreading import WattsCascadesSystem
        >>> WattsCascadesSystem(nx.cycle_graph(10), random_state=0).prepare().system_.elements[0].size
        10
    """

    field = 'social/spreading'
    topics = ('cascades', 'agents', 'network', 'threshold')
    references = ('Watts 2002', 'Pastor-Satorras et al. 2015')
    until = 30

    def __init__(
        self,
        graph: object = None,
        *,
        threshold: float = 0.18,
        seeds: int = 1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.graph = graph
        self.threshold = threshold
        self.seeds = seeds
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.gnp_random_graph(200, 4 / 199, seed=int(rng.integers(2**31))) if graph is None else graph
        size = nx.number_of_nodes(graph)
        active = np.zeros(size)
        active[rng.choice(size, self.seeds, replace=False)] = 1.0
        nodes = ThresholdNodeElement('nodes', active=active, threshold=self.threshold, size=size)
        self.system_ = System(
            self.label,
            elements=[nodes],
            interactions=[ThresholdActivationInteraction('activation', target='nodes')],
            space=NetworkSpace(graph),
        )
