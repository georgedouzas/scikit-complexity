"""Define bounded confidence coupling, the move of every agent to the mean opinion of the neighbours it trusts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import CouplingInteraction
from ....core.laws.utils import where
from ....core.spaces import AdjacencyNeighborhood
from ....specs import RoleSpec


class ConfidenceCouplingInteraction(CouplingInteraction):
    """Hegselmann-Krause coupling, the move of every agent to the mean opinion of the neighbours within its confidence.

    Every step every agent takes at once the mean opinion of its neighbours, itself included, whose opinions differ
    from its own by less than its parameter `confidence`. It is the map form of coupling with the normalized mean over
    the neighbours of the network weighted by the indicator of the confidence.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of agents, on the nodes of a network.

        state:
            The name of the state that holds the opinions.

    Examples:
        >>> import networkx as nx
        >>> import numpy as np
        >>> from skcomplex.core.elements import LatticeElement
        >>> from skcomplex.core.spaces import DiscreteTime, NetworkSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.social.opinion_dynamics import ConfidenceCouplingInteraction, ConfidentElement
        >>> people = LatticeElement(
        ...     'people',
        ...     element=ConfidentElement,
        ...     space=NetworkSpace(nx.complete_graph(3)),
        ...     values={'opinion': [0.1, 0.2, 0.9], 'confidence': 0.2},
        ... )
        >>> coupling = ConfidenceCouplingInteraction('compromise', target='people')
        >>> system = System(elements=[people], interactions=[coupling], time=DiscreteTime())
        >>> opinions = Simulation(system).check().prepare().run(until=1).trajectory_.values('people.opinion')
        >>> np.round(opinions[-1], 6).tolist()
        [0.15, 0.15, 0.9]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], state: str = 'opinion') -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._get_opinion,
            reduce='normalized-mean',
            weights=self._get_trust,
            form='map',
            neighborhood=AdjacencyNeighborhood(center=True),
        )
        self.state = state

    def _get_opinion(self, namespace: SimpleNamespace) -> Expression:
        """Return the opinion of the neighbour."""
        return getattr(namespace.other, self.state)

    def _get_trust(self, namespace: SimpleNamespace) -> Expression:
        """Return 1 for a neighbour whose opinion lies within the confidence of the agent, and 0 otherwise."""
        difference = getattr(namespace.other, self.state) - getattr(namespace.member, self.state)
        return where(abs(difference) < namespace.member.confidence, 1, 0)
