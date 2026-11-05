"""FitzHugh-Nagumo neuron, with a fast potential `v` and a slow recovery `w`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class FitzHughNagumoNeuronElement(Element):
    """FitzHugh-Nagumo neuron, with a fast potential `v` and a slow recovery `w`.

    The neuron has no laws of its own. FitzHugh-Nagumo kinetics change it.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        w:
            The initial recovery.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import FitzHughNagumoNeuronElement
        >>> FitzHughNagumoNeuronElement('neuron').w
        -0.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -1.2,
        w: Annotated[float, StateSpec()] = -0.6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.w = w
