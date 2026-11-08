"""Define edge draws of the weights of a network.

An edge draw draws the weight of every new edge of a network and lets the other weights drift.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import NeighborInteraction
from .....core.components.laws import BaseLaw, UpdateLaw
from .....core.components.laws.utils import normal, uniform, where
from .....core.specs import ParameterSpec, RoleSpec


class EdgeDrawInteraction(NeighborInteraction):
    """First draw of the weight of every new edge of a network.

    The weights of the other edges drift at random.

    Every step of discrete time an edge marked `fresh` draws its weight uniformly from `[low, high]` and loses its mark,
    and every other edge adds `noise` times a standard normal draw to its weight, or keeps it without noise.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group on the nodes of the network.

        edges:
            The path of the group on its edges, whose members hold the weight and the mark.

        noise:
            The standard deviation of the drift of a weight in a step, 0 for none.

        low:
            The least weight of a new edge.

        high:
            The largest weight of a new edge.

        state:
            The name of the state of the edges that holds the weight.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.biology.common.networks import EdgeDrawInteraction
        >>> edge = SimpleNamespace(w=var('w'), fresh=var('fresh'))
        >>> draw = EdgeDrawInteraction('draw', target='species', edges='links').prepare()
        >>> [str(law.value) for law in draw.run(SimpleNamespace(edge=edge, noise=var('noise'))).laws_][1]
        '0'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        edges: Annotated[str, RoleSpec()],
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        low: float = -1.0,
        high: float = 1.0,
        state: str = 'w',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.edges = edges
        self.noise = noise
        self.low = low
        self.high = high
        self.state = state

    def _get_name(self, draw: str) -> str:
        """Return the name of a random draw of the interaction."""
        return re.sub(r'\W', '_', f'{draw}_{self.label}')

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the draw or the drift of the weight of an edge.

        The removal of the mark of the edge comes with it.

        Args:
            namespace:
                The namespace of an edge, with the member `namespace.edge` of the group on the edges.

        Returns:
            laws:
                The updates of the weight and of the mark.
        """
        edge = namespace.edge
        weight = getattr(edge, self.state)
        drawn = uniform(self._get_name('draw'), self.low, self.high)
        kept = weight + namespace.noise * normal(self._get_name('drift')) if self.noise else weight
        return [UpdateLaw(weight, where(edge.fresh > 1 / 2, drawn, kept)), UpdateLaw(edge.fresh, 0)]
