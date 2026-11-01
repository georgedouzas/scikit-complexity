"""Define logistic map coupling, which mixes the logistic map of every node with the mean map of its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import CouplingInteraction
from ....core.types import Target
from ....specs import RoleSpec


class LogisticCouplingInteraction(CouplingInteraction):
    """Coupled logistic maps, `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)` with `f(x) = r x (1 - x)`.

    The map of every node mixes with the mean of the maps of its neighbours `j` on its lattice or network, with the
    parameters `r` and `epsilon` of the node. A node with no neighbours keeps `(1 - epsilon) f(x_i)`. It is the map
    form of coupling, with the mean of the maps of the neighbours combined with the map of the node.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of maps, on the nodes of a lattice or network.

        state:
            The name of the state that the maps iterate.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.elements import LatticeElement
        >>> from skcomplex.core.spaces import DiscreteTime, NetworkSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.physics.synchronization import LogisticCouplingInteraction, LogisticNodeElement
        >>> maps = LatticeElement(
        ...     'maps',
        ...     element=LogisticNodeElement,
        ...     space=NetworkSpace(nx.path_graph(2)),
        ...     values={'x': [0.25, 0.5], 'epsilon': 0.5},
        ... )
        >>> coupling = LogisticCouplingInteraction('coupling', target='maps')
        >>> system = System(elements=[maps], interactions=[coupling], time=DiscreteTime())
        >>> Simulation(system).check().prepare().run(until=1).trajectory_.values('maps.x')[-1].tolist()
        [0.875, 0.875]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], state: str = 'x') -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._get_map,
            reduce='mean',
            form='map',
            combine=self._get_mix,
        )
        self.state = state

    @staticmethod
    def _logistic(r: Expression, x: Expression) -> Expression:
        """Return the logistic map `r x (1 - x)`."""
        return r * x * (1 - x)

    def _get_map(self, namespace: SimpleNamespace) -> Expression:
        """Return the map of the neighbour, with the growth rate of the node."""
        return self._logistic(namespace.member.r, getattr(namespace.other, self.state))

    def _get_mix(self, namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the map of the node mixed with the mean map of its neighbours."""
        member = namespace.member
        own = self._logistic(member.r, getattr(member, self.state))
        return (1 - member.epsilon) * own + member.epsilon * reduction
