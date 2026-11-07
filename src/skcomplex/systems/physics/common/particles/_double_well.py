"""Particle at the position `x` in the double well `x^4 / 4 - x^2 / 2`, forced periodically and by noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class DoubleWellElement(Element):
    """Particle at the position `x` in the double well `x^4 / 4 - x^2 / 2`, forced periodically and by noise.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for none.

        x:
            The initial position.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import DoubleWellElement
        >>> DoubleWellElement('particle', D=0.2).D
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = -1.0,
        amplitude: Annotated[float, ParameterSpec()] = 0.1,
        frequency: Annotated[float, ParameterSpec()] = 0.01,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.12,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.amplitude = amplitude
        self.frequency = frequency
        self.D = D
