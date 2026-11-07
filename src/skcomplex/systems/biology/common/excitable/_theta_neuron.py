"""Theta neuron of Ermentrout and Kopell, `theta' = (1 - cos theta) + (1 + cos theta) eta`, a spiking phase."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, pi

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, RateLaw
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class ThetaNeuronElement(Element):
    """Theta neuron of Ermentrout and Kopell, `theta' = (1 - cos theta) + (1 + cos theta) eta`, a spiking phase.

    The neuron fires when its phase passes π. With `eta > 0` it fires periodically with period `π / sqrt(eta)`, and
    with `eta < 0` it is excitable. The change of variable `V = tan(theta / 2)` turns it into the quadratic
    integrate-and-fire neuron `V' = V² + eta`.

    Args:
        label:
            The label of the neuron or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        eta:
            The excitability.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.neuroscience import ThetaNeuronElement
        >>> GroupElement('neurons', element=ThetaNeuronElement, size=10, values={'eta': 0.5}).size
        10
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.eta = eta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the phase dynamics.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.theta, 1 - cos(s.theta) + (1 + cos(s.theta)) * s.eta)]
