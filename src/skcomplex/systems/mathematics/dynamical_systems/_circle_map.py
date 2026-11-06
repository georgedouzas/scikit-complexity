"""Angle and parameters of the sine circle map `theta' = theta + Omega - K / (2 pi) sin(2 pi theta) (mod 1)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CircleMapElement(Element):
    """Angle and parameters of the sine circle map `theta' = theta + Omega - K / (2 pi) sin(2 pi theta) (mod 1)`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial angle, as a fraction of a turn.

        Omega:
            The bare rotation per step.

        K:
            The nonlinearity.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CircleMapElement
        >>> CircleMapElement('map', K=1.0).K
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        Omega: Annotated[object, ParameterSpec()] = 0.25,
        K: Annotated[object, ParameterSpec()] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.Omega = Omega
        self.K = K
