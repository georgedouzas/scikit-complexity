"""Define sited oscillators, phase oscillators with a place in a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from .....core.specs import StateSpec, VisualSpec
from ._phase_oscillator import PhaseOscillatorElement


class SitedOscillatorElement(PhaseOscillatorElement):
    """Phase oscillator with a place `x` in a continuous space, which couplings within a distance or swarming read.

    The phase turns at the natural frequency, and the place changes only by the rates of interactions.

    Args:
        label:
            The label of the oscillator or the group.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        omega:
            The natural frequency.

        x:
            The initial place. A number applies to every component.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.common.oscillators import SitedOscillatorElement
        >>> SitedOscillatorElement('unit', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: float = 0.0,
        omega: float = 1.0,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='site')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, theta=theta, omega=omega)
        self.x = x
