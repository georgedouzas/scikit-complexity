"""Walker that jumps at every step by a length of power-law tail `l^(-1 - mu)` in a random direction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._continuous_time_walker import POINTS


class LevyFlierElement(Element):
    """Walker that jumps at every step by a length of power-law tail `l^(-1 - mu)` in a random direction.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        mu:
            The tail exponent of the jump lengths, below 2 for jumps of infinite variance.

        l0:
            The shortest jump length.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyFlierElement
        >>> LevyFlierElement('walker', mu=1.2).mu
        1.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        l0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.mu = mu
        self.l0 = l0
