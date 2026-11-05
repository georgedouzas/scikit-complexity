"""Define the hidden units of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class HiddenUnitElement(Element):
    """Hidden unit of a simple recurrent network, whose activity is a sigmoid of its recurrent input and its drive.

    The drive `u` holds the weighted input of the previous step, and the bias `b` is adaptive, so training changes it.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        s:
            The activity in `(0, 1)`.

        u:
            The weighted input of the previous step.

        b:
            The bias, an adaptive parameter.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HiddenUnitElement
        >>> HiddenUnitElement('unit').s
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec()] = 0.5,
        u: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.u = u
        self.b = b
