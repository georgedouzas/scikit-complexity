"""Light body at the height `z`, of order 2, on the axis through the centre of mass of two equal primaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SitnikovBodyElement(Element):
    """Light body at the height `z`, of order 2, on the axis through the centre of mass of two equal primaries.

    The primaries orbit with the eccentricity `e`.

    Args:
        label:
            The label of the body.

        visual:
            How the body is drawn, or `None` for none.

        z:
            The initial height.

        z_d1:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import SitnikovBodyElement
        >>> SitnikovBodyElement('body', e=0.1).e
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(order=2)] = 0.5,
        z_d1: float = 0.0,
        e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.z_d1 = z_d1
        self.e = e
