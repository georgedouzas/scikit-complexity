"""Local wave number of deep-water waves, conserved as `dk/dt + d omega(k)/dx = 0` with `omega = sqrt(g k)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WaveNumberElement(Element):
    """Local wave number of deep-water waves, conserved as `dk/dt + d omega(k)/dx = 0` with `omega = sqrt(g k)`.

    The wave number travels at the group speed `C(k) = sqrt(g / k) / 2`, which falls as the waves shorten, so long
    waves catch up with short ones and the wave number steepens until it breaks.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        k:
            The initial wave number.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.models.physics.fluids import WaveNumberElement
        >>> WaveNumberElement('site').g
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k
        self.g = g
