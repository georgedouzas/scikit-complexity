"""Site of the Carleman discrete-velocity gas, densities `f_plus` and `f_minus` moving at the speeds `c` and `-c`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CarlemanGasElement(Element):
    """Site of the Carleman discrete-velocity gas, densities `f_plus` and `f_minus` moving at the speeds `c` and `-c`.

    Collisions turn two particles of one velocity into two of the other at the rate `kappa`, so
    `df_plus/dt + c df_plus/dx = kappa (f_minus^2 - f_plus^2)` and the opposite for `f_minus`, which conserves the
    mass and relaxes to the equilibrium `f_plus = f_minus`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        f_plus:
            The initial density of right-moving particles.

        f_minus:
            The initial density of left-moving particles.

        c:
            The particle speed.

        kappa:
            The collision rate.

    Examples:
        >>> from skcomplex.models.physics.fluids import CarlemanGasElement
        >>> CarlemanGasElement('site').kappa
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f_plus: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        f_minus: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f_plus = f_plus
        self.f_minus = f_minus
        self.c = c
        self.kappa = kappa
