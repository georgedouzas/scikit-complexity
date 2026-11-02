"""Walker that waits a power-law time `tau0 r^(-1 / alpha)` for `r` uniform, then jumps by a Gaussian step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec

POINTS = VisualSpec('points', {'position': 'x'})


class ContinuousTimeWalkerElement(Element):
    """Walker that waits a power-law time `tau0 r^(-1 / alpha)` for `r` uniform, then jumps by a Gaussian step.

    The jumps are a jump interaction of the model, which resets the position after every waiting time.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        alpha:
            The tail exponent of the waiting times, below 1 for waiting times without a mean.

        tau0:
            The shortest waiting time.

        sigma:
            The standard deviation of the jumps along every axis.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import ContinuousTimeWalkerElement
        >>> ContinuousTimeWalkerElement('walker', alpha=0.7).alpha
        0.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        tau0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.tau0 = tau0
        self.sigma = sigma
