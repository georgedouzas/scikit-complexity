"""Transverse position and momentum of a particle of an accelerator beam, kicked by the opposite beam every turn."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BeamElement(Element):
    """Transverse position and momentum of a particle of an accelerator beam, kicked by the opposite beam every turn.

    Map interactions rotate the particle by the tune and kick its momentum with the field of the opposite beam, so the
    element holds the states and the parameters they read.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial transverse position.

        y:
            The initial transverse momentum.

        nu:
            The tune, the rotation per turn in units of `pi`, in `[0, 2]`.

        c:
            The strength of the beam-beam kick.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import BeamElement
        >>> BeamElement('particle').nu
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        y: Annotated[float, StateSpec()] = 0.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 2))] = 0.3,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.nu = nu
        self.c = c
