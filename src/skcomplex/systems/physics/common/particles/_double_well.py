"""Define particles in the double well `x^4 / 4 - x^2 / 2`.

The particles are forced periodically and by noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, sqrt
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.types import Role


class DoubleWellElement(Element):
    """Particles at the positions `x` in a double well.

    Args:
        label:
            The label of the particles.

        x:
            The initial position, one per particle or shared.

        size:
            The number of particles, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import DoubleWellElement
        >>> DoubleWellElement('particle', x=1.0).x
        1.0
    """

    def __init__(self, label: str, *, x: object = -1.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class DoubleWellInteraction(Interaction):
    """Overdamped motion in the double well `x^4 / 4 - x^2 / 2`, forced periodically and by noise.

    The position follows `dx = (x - x^3 + amplitude cos(2 pi frequency t)) dt + sqrt(2 D) dW`. It is a particle that
    hops between two wells, a bistable switch, or the climate between two states.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.particles import DoubleWellElement, DoubleWellInteraction
        >>> well = DoubleWellInteraction('well', target='particle', amplitude=0.0, D=0.0)
        >>> system = System(elements=[DoubleWellElement('particle', x=0.5)], interactions=[well]).prepare()
        >>> round(float(system.simulate(until=20.0, dt=1.0).get_values('particle.x')[-1]), 3)
        1.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        amplitude: float = 0.1,
        frequency: float = 0.01,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 0.12,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.amplitude = amplitude
        self.frequency = frequency
        self.D = D

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drift and the noise of the position of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        x = s.target.x
        drive = s.amplitude * cos(2 * math.pi * s.frequency * t)
        self.laws_ = [RateLaw(x.diff(t), x - x**3 + drive), NoiseLaw(x, sqrt(2 * s.D))]
