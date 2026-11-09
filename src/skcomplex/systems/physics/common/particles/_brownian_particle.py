"""Overdamped Brownian particle at the position `x`.

The particle has the friction `gamma` and the thermal energy `temperature`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec, StateSpec, VisualSpec

POINTS = VisualSpec("points", {"position": "x", "radius": 0.05})


class BrownianParticleElement(Element):
    """Overdamped Brownian particle at the position `x`.

    The particle has the friction `gamma` and the thermal energy `temperature`.

    The particle moves by `dx = F / gamma dt + sqrt(2 temperature / gamma) dW`, with the thermal noise of
    `ThermalNoiseInteraction` and the forces of other interactions, which divide by `gamma`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        gamma:
            The friction coefficient.

        temperature:
            The thermal energy.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import BrownianParticleElement
        >>> BrownianParticleElement('bead', temperature=2.0).temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role="position")] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.gamma = gamma
        self.temperature = temperature
