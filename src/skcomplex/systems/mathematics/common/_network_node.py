"""Node of a network that carries its degree, clustering, betweenness, and community."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import StateSpec, VisualSpec


class NetworkNodeElement(Element):
    """Node of a network that carries its degree, clustering, betweenness, and community.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The degree of the node.

        clustering:
            The clustering coefficient of the node.

        betweenness:
            The normalized betweenness centrality of the node.

        community:
            The index of the community of the node.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.networks import NetworkNodeElement
        >>> group = GroupElement('nodes', element=NetworkNodeElement, size=3)
        >>> group.label, group.element.__name__, group.size
        ('nodes', 'NetworkNodeElement', 3)
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'community'}),
        *,
        degree: Annotated[float, StateSpec(domain=NN)] = 0.0,
        clustering: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        betweenness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        community: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.degree = degree
        self.clustering = clustering
        self.betweenness = betweenness
        self.community = community

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that keeps the node in place.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.stay)]

    def stay(self, agent: AgentView, system: SystemView) -> None:
        """Keep the node and its measures unchanged.

        Args:
            agent:
                The node.

            system:
                The system.
        """
