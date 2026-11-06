"""Define the output units of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class OutputUnitElement(Element):
    """Output unit of a simple recurrent network, whose activity predicts a target.

    The bias `c` is adaptive, so training changes it, and the target is a parameter a dataset or a schedule sets.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        y:
            The prediction in `(0, 1)`.

        c:
            The bias, an adaptive parameter.

        target:
            The value the prediction is compared with.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import OutputUnitElement
        >>> OutputUnitElement('prediction').target
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = 0.5,
        c: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
        target: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.c = c
        self.target = target
