"""Particle at the position `x` in a box `[0, L]` whose walls reflect it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, BoundaryLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ModulatedParticleElement(Element):
    """Particle at the position `x` in a box `[0, L]` whose walls reflect it.

    The noise amplitude `sigma = s0 + ds (x / L - 1/2)` grows across the box. The walls reflect the particle at every
    frame, so frames must be short against the time to cross the box.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        s0:
            The noise amplitude at the middle of the box.

        ds:
            The change of the noise amplitude across the box.

        L:
            The length of the box.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import ModulatedParticleElement
        >>> ModulatedParticleElement('bead').ds
        1.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 50.0,
        s0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        ds: Annotated[float, ParameterSpec()] = 1.8,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.s0 = s0
        self.ds = ds
        self.L = L

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the reflection of the particle at the walls of the box.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The boundary law of the position.
        """
        return [BoundaryLaw(s.x, where(s.x < 0, -s.x, where(s.x > s.L, 2 * s.L - s.x, s.x)))]
