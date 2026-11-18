"""Define sited oscillators, phase oscillators with a place in a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping
from typing import Annotated

from .....core.components.elements import Period
from .....core.components.spaces import BaseSpace
from ._phase_oscillator import PhaseOscillatorElement


class SitedOscillatorElement(PhaseOscillatorElement):
    """Phase oscillator with a place `x` in a continuous space.

    The place is named after the axis of the space, so it is the position of the oscillator, which couplings within a
    distance read. The phase turns at the natural frequency, and the place changes only by the rates of interactions.

    Args:
        label:
            The label of the oscillator or the group.

        theta:
            The initial phase.

        omega:
            The natural frequency.

        x:
            The initial place, one row per member or shared.

        size:
            The number of oscillators, or `None` for one.

        space:
            The continuous space of the oscillators, or `None` for none.

        values:
            Further values by name, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.common.oscillators import SitedOscillatorElement
        >>> SitedOscillatorElement('unit', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        omega: object = 1.0,
        x: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, theta=theta, omega=omega, size=size, space=space, values=values)
        self.x = x
