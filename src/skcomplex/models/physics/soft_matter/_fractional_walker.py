"""Walker driven by fractional Brownian motion, `dx = sqrt(2 D) dB_H`, whose increments are correlated in time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._continuous_time_walker import POINTS


class FractionalWalkerElement(Element):
    """Walker driven by fractional Brownian motion, `dx = sqrt(2 D) dB_H`, whose increments are correlated in time.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        D:
            The diffusion coefficient.

        H:
            The Hurst index of the driving fractional Brownian motion.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import FractionalWalkerElement
        >>> FractionalWalkerElement('walker', H=0.3).H
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        H: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D
        self.H = H
