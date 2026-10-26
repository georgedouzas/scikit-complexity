"""Define point particles, elements with a position, a velocity, and a mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy.typing as npt

from ...core.base import BaseLaw, NumericDomain, Requires, SpaceKind, VisualSpec, parameter, state
from ...core.components import Element
from ...core.laws import RateLaw, Symbols


class ParticleElement(Element):
    """Point particle with a position, a velocity, and a mass.

    The particle moves with its velocity, and every force on it changes its velocity by the force divided by its mass.

    Args:
        label:
            The label of the particle.

        x:
            The initial position. A number applies to every component.

        v:
            The initial velocity. A number applies to every component.

        m:
            The mass.

    Examples:
        >>> from skcomplex.formalisms.particles import ParticleElement
        >>> ParticleElement('ball', x=[0, 1], m=2.0)
        ParticleElement(label='ball', m=2.0, v=0.0, x=[0, 1])
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.CONTINUOUS})

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x'})
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    v: npt.ArrayLike = state(default=0.0, vector=True, velocity_of='x')
    m: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the kinematic law `dx/dt = v`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, s.v)]
