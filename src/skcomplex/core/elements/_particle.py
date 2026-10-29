"""Define particles, point elements with a position, a velocity, and a mass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, Self

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...specs import ParameterSpec, StateSpec, VisualSpec
from ..laws import BaseLaw, RateLaw
from ..spaces import BaseSpace
from ._element import Element


class ParticleElement(Element):
    """Point particle with a position, a velocity, and a mass, in a continuous space.

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

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            Points at the positions of the particle.

    Examples:
        >>> from skcomplex.core.elements import ParticleElement
        >>> ParticleElement('ball', x=[0, 1], m=2.0).check().prepare().default_visual_.kind
        'points'
    """

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, position=True)] = 0.0,
        v: Annotated[npt.ArrayLike, StateSpec(vector=True, velocity_of='x')] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m

    def _prepare(self, context: BaseSpace | None = None) -> Self:
        """Set `default_visual_`, points at the positions of the particle.

        Args:
            context:
                The space the element inherits from its system, which the element does not read.

        Returns:
            element:
                The element itself.
        """
        self.default_visual_ = VisualSpec('points', {'position': 'x'})
        return self

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the kinematic law `dx/dt = v`.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the particle.
        """
        return [RateLaw(namespace.x, namespace.v)]
