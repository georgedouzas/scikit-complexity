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

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, UpdateLaw
from .....core.components.laws.utils import normal, uniform, where
from .....core.types import Role


class EdgeDrawInteraction(Interaction):
    """First draw of the weight of every new edge of a network.

    The weights of the other edges drift at random.

    Every step of discrete time an edge marked `fresh` draws its weight uniformly from `[low, high]` and loses its mark,
    and every other edge adds `noise` times a standard normal draw to its weight, or keeps it without noise.

    Args:
        label:
            The label of the interaction.

        among:
            The path of the group on the nodes of the network. The weight and the mark are values of the group on its
            edges.

        noise:
            The standard deviation of the drift of a weight in a step, 0 for none.

        low:
            The least weight of a new edge.

        high:
            The largest weight of a new edge.

        state:
            The name of the state of the edges that holds the weight.

        along:
            The edges the laws run on, `True` for the edges of the network the group lives on, or the name of a layer.

    Examples:
        >>> from skcomplex.systems.biology.common.networks import EdgeDrawInteraction
        >>> EdgeDrawInteraction('draw', among='species', noise=0.1).noise
        0.1
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        noise: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        low: float = -1.0,
        high: float = 1.0,
        state: str = 'w',
        along: bool | str = True,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.noise = noise
        self.low = low
        self.high = high
        self.state = state

    def _get_name(self, draw: str) -> str:
        """Return the name of a random draw of the interaction."""
        return re.sub(r'\W', '_', f'{draw}_{self.label}')

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the draw or the drift of the weight of an edge, and the removal of the mark of the edge.

        Args:
            context:
                The namespace of an edge, with the member `context.edge` of the group on the edges.
        """
        super()._prepare(context)
        edge = context.edge
        weight = getattr(edge, self.state)
        drawn = uniform(self._get_name('draw'), self.low, self.high)
        kept = weight + context.noise * normal(self._get_name('drift')) if self.noise else weight
        self.laws_: list[BaseLaw] = [
            UpdateLaw(weight, where(edge.fresh > 1 / 2, drawn, kept)),
            UpdateLaw(edge.fresh, 0),
        ]
