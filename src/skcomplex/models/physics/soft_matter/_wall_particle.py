"""Brownian particle at the distance `z` from a wall, which reflects it, and whose diffusion slows near the wall."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, BoundaryLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WallParticleElement(Element):
    """Brownian particle at the distance `z` from a wall, which reflects it, and whose diffusion slows near the wall.

    Args:
        label:
            The label of the particle.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The distance from the wall.

        D0:
            The diffusion coefficient far from the wall.

        a:
            The distance over which the wall slows diffusion.

        drift:
            The settling velocity far from the wall.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import WallParticleElement
        >>> WallParticleElement('bead').a
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        drift: Annotated[float, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.D0 = D0
        self.a = a
        self.drift = drift

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reflection at the wall.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [BoundaryLaw(s.z, abs(s.z))]
