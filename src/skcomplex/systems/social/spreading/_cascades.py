"""Cascade systems: adoption that spreads when enough neighbours have adopted."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class ThresholdNodeView(AgentView):
    """View of a node of the Watts cascade system, with its states and parameters as attributes."""

    active: float
    threshold: float


class ThresholdActivationInteraction(MapInteraction):
    """Activation of the Watts cascade system, where a node becomes active once enough of its neighbours are active.

    A node becomes active for good once the share of active nodes among its neighbouring nodes reaches its threshold.
    The nodes update one after the other in a new random order every step, each reading the activations the nodes
    before it set, so a cascade can cross several nodes in one step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of nodes.

    Examples:
        >>> from skcomplex.systems.social.spreading import ThresholdActivationInteraction
        >>> ThresholdActivationInteraction('activation', target='nodes').state
        'active'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='active',
            function=self._get_active,
            schedule='random-sequential',
            form='agent',
        )

    def _get_active(self, agent: ThresholdNodeView, system: SystemViewProtocol) -> float:
        """Return 1 for a node that is or becomes active, and 0 for one that stays inactive."""
        del system
        neighbors = [item for item in agent.neighbors() if isinstance(item, ThresholdNodeView)]
        if agent.active or not neighbors:
            return float(agent.active)
        return float(sum(neighbor.active for neighbor in neighbors) / len(neighbors) >= agent.threshold)


class ThresholdNodeElement(Element):
    """Node of the Watts cascade system, which the activation turns active once enough of its neighbours are active.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        active:
            Whether the node starts active.

        threshold:
            The share of active neighbours at which the node becomes active.

    Examples:
        >>> from skcomplex.systems.social.spreading import ThresholdNodeElement
        >>> group = GroupElement('nodes', element=ThresholdNodeElement, size=10, values={'threshold': 0.18})
        >>> group.label, group.element.__name__, group.size
        ('nodes', 'ThresholdNodeElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'active', 'color_range': (0.0, 1.0)}),
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
    ) -> None:
        super().__init__(label, visual=visual)
        self.active = active
        self.threshold = threshold

    @classmethod
    def _get_view(cls) -> type[ThresholdNodeView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ThresholdNodeView


@register_system(
    field='social/spreading',
    tags=('cascades', 'agents', 'network', 'threshold'),
    references=('Watts 2002', 'Pastor-Satorras et al. 2015'),
    defaults={'until': 30},
)
def watts_cascades(
    graph: object = None,
    *,
    threshold: float = 0.18,
    seeds: int = 1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Watts threshold system of global cascades started by a few random seeds.

    Args:
        graph:
            The NetworkX graph, or `None` for a random graph of 200 nodes with mean degree 4.

        threshold:
            The share of active neighbours at which a node becomes active.

        seeds:
            The number of nodes active at the start.

        random_state:
            The seed or generator of the default graph and of the seeds.

    Returns:
        system:
            The system, with the group `nodes`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.social.spreading import watts_cascades
        >>> watts_cascades(nx.cycle_graph(10), random_state=0).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(random_state)
    graph = nx.gnp_random_graph(200, 4 / 199, seed=int(rng.integers(2**31))) if graph is None else graph
    size = nx.number_of_nodes(graph)
    active = np.zeros(size)
    active[rng.choice(size, seeds, replace=False)] = 1.0
    nodes = GroupElement(
        'nodes',
        element=ThresholdNodeElement,
        size=size,
        values={'active': active, 'threshold': threshold},
    )
    return System(
        elements=[nodes],
        interactions=[ThresholdActivationInteraction('activation', target='nodes')],
        space=NetworkSpace(graph),
    )
