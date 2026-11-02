"""Site of a string, with the transverse displacement `u` of order 2 and the wave speed `c`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WaveStringElement(Element):
    """Site of a string, with the transverse displacement `u` of order 2 and the wave speed `c`.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial displacement.

        u_d1:
            The initial transverse velocity.

        c:
            The wave speed.

    Examples:
        >>> from skcomplex.models.physics.waves import WaveStringElement
        >>> WaveStringElement('site', c=2.0).c
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(order=2)] = 0.0,
        u_d1: float = 0.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.u_d1 = u_d1
        self.c = c
