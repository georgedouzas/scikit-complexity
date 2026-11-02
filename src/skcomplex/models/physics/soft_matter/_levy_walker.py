"""Walker that flies at constant speed in a random direction for a power-law flight time, then turns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._continuous_time_walker import POINTS


class LevyWalkerElement(Element):
    """Walker that flies at constant speed in a random direction for a power-law flight time, then turns.

    The flight is a rate interaction of the position, and the turns a jump interaction that resets the velocity after
    every flight time.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        v:
            The initial velocity.

        sigma:
            The tail exponent of the flight times, between 1 and 2 for superdiffusion.

        tau0:
            The shortest flight time.

        speed:
            The speed of every flight.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import LevyWalkerElement
        >>> LevyWalkerElement('walker', speed=2.0).speed
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        v: Annotated[npt.ArrayLike, StateSpec(vector=True, role='velocity')] = 0.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        tau0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.sigma = sigma
        self.tau0 = tau0
        self.speed = speed
