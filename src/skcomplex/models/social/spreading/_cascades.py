"""Cascade models: adoption that spreads when enough neighbours have adopted."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import NetworkSpace
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ....specs import StateSpec, VisualSpec
from ..._catalog import register_model


class ThresholdNodeElement(Element):
    """Node of the Watts cascade model, which becomes active once the share of active neighbours reaches its threshold.

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
        >>> from skcomplex.models.social.spreading import ThresholdNodeElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the activation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.activate)]

    def activate(self, agent: AgentView, model: ModelView) -> None:
        """Become active once the share of active neighbours reaches the threshold.

        Args:
            agent:
                The node.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if agent.active or not neighbors:
            return
        if sum(neighbor.active for neighbor in neighbors) / len(neighbors) >= agent.threshold:
            agent.active = 1.0


@register_model(
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
    """Return the Watts threshold model of global cascades started by a few random seeds.

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
            The model, with the group `nodes`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.spreading import watts_cascades
        >>> watts_cascades(nx.cycle_graph(10), random_state=0).elements[0].check().prepare().size_
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
    return System(elements=[nodes], space=NetworkSpace(graph))
