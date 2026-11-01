"""Angle, momentum, and kick of the standard map `p' = p + K sin(theta)`, `theta' = theta + p' (mod 2 pi)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2


class StandardMapElement(Element):
    """Angle, momentum, and kick of the standard map `p' = p + K sin(theta)`, `theta' = theta + p' (mod 2 pi)`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial angle.

        p:
            The initial momentum.

        K:
            The kick strength.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import StandardMapElement
        >>> StandardMapElement('map', K=0.5).K
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        p: Annotated[object, StateSpec()] = 2 * math.pi * GOLDEN_MEAN,
        K: Annotated[object, ParameterSpec()] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.p = p
        self.K = K
