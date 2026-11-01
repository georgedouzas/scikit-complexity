"""Site of a channel of shallow water, with the depth `h` and the discharge `q` per unit width."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ShallowWaterElement(Element):
    """Site of a channel of shallow water, with the depth `h` and the discharge `q` per unit width.

    The mass `h` and the momentum `q` are conserved by two conservation interactions, with the fluxes `q` and
    `q^2 / h + g h^2 / 2`, which form one hyperbolic system.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        h:
            The initial depth.

        q:
            The initial discharge.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.models.physics.fluids import ShallowWaterElement
        >>> ShallowWaterElement('site', h=2.0).h
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        q: Annotated[float, StateSpec()] = 0.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.q = q
        self.g = g
