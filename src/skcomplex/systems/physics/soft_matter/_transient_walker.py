"""Brownian walker whose diffusion coefficient `D` is redrawn after waiting times of mean `D^(-gamma)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TransientWalkerElement(Element):
    """Brownian walker whose diffusion coefficient `D` is redrawn after waiting times of mean `D^(-gamma)`.

    The coefficient takes values of density `sigma D^(sigma - 1)` on `(0, 1)`, the beta distribution `B(sigma, 1)`, so
    slow walkers wait long before they change.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        D:
            The initial diffusion coefficient.

        sigma:
            The exponent of the density of diffusion coefficients near zero.

        gamma:
            The exponent of the mean waiting time `D^(-gamma)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TransientWalkerElement
        >>> TransientWalkerElement('walker', gamma=2.0).gamma
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('points', {'position': 'x', 'color_by': 'D', 'color_range': (0.0, 1.0)}),
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        D: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D
        self.sigma = sigma
        self.gamma = gamma
