"""Theta neuron of Ermentrout and Kopell, a spiking phase.

The phase follows `theta' = (1 - cos theta) + (1 + cos theta) eta`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from .....core.components.elements import Element, Period
from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Array, Role


class ThetaNeuronElement(Element):
    """Theta neurons, each with a phase `theta` that wraps around and an excitability `eta`.

    Args:
        label:
            The label of the neuron or the group.

        theta:
            The initial phase, one per neuron or shared.

        eta:
            The excitability, one per neuron or shared.

        size:
            The number of neurons, or `None` for one.

        space:
            The space whose sites hold one neuron each, such as a network, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ThetaNeuronElement
        >>> ThetaNeuronElement('neurons', eta=0.5, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float | Array, Period(2 * math.pi)] = 0.0,
        eta: float | Array = 1.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.theta = theta
        self.eta = eta


class ThetaNeuronInteraction(Interaction):
    """Phase dynamics of theta neurons, `theta' = (1 - cos theta) + (1 + cos theta) eta`.

    The neuron fires when its phase passes π. With `eta > 0` it fires periodically with period `π / sqrt(eta)`, and
    with `eta < 0` it is excitable. The change of variable `V = tan(theta / 2)` turns it into the quadratic
    integrate-and-fire neuron `V' = V² + eta`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the neurons, whose values are `theta` and `eta`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import ThetaNeuronInteraction
        >>> ThetaNeuronInteraction('phases', target='neurons').target
        'neurons'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the phase dynamics of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            theta, eta = context.target.theta, context.target.eta
            self.laws_ = [RateLaw(theta, 1 - cos(theta) + (1 + cos(theta)) * eta)]
