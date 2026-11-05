"""Define swarmalators, phase oscillators that move in a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.specs import StateSpec, VisualSpec
from ..common.oscillators import SitedOscillatorElement


class SwarmalatorElement(SitedOscillatorElement):
    """Phase oscillator whose place `x` is a position that moves by the rates of interactions.

    Args:
        label:
            The label of the swarmalator or the group.

        visual:
            How the swarmalator is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        omega:
            The natural frequency.

        x:
            The initial position. A number applies to every component.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorElement
        >>> SwarmalatorElement('unit', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: float = 0.0,
        omega: float = 1.0,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, theta=theta, omega=omega, x=x)
