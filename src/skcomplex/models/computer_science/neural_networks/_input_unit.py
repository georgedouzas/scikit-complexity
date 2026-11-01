"""Define the input units of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, VisualSpec


class InputUnitElement(Element):
    """Input unit of a simple recurrent network, whose value a dataset or a schedule sets at every step.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The input.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import InputUnitElement
        >>> InputUnitElement('bit').x
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
