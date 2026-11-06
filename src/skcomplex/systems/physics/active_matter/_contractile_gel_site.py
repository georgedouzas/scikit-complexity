"""Site of a one-dimensional active viscoelastic gel, the density of contractile cells and the gel displacement."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ContractileGelSiteElement(Element):
    """Site of a one-dimensional active viscoelastic gel, the density of contractile cells and the gel displacement.

    The site holds the density `c` of the cells and the displacement `u` of a Kelvin-Voigt gel of stiffness `E` and
    viscosity `eta`, with friction `gamma` against a rigid substrate. The cells diffuse with the coefficient `D` and
    exert the active stress `T0 c / (c + c_sat)`.

    Args:
        label:
            The label of the site or the gel.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        c:
            The initial density of the cells.

        u:
            The initial displacement of the gel.

        D:
            The diffusion coefficient of the cells.

        gamma:
            The friction of the gel with the substrate.

        eta:
            The viscosity of the gel.

        E:
            The stiffness of the gel.

        T0:
            The largest active stress of the cells.

        c_sat:
            The density at which the active stress is half its largest value.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ContractileGelSiteElement
        >>> ContractileGelSiteElement('site').T0
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        E: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        T0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        c_sat: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.u = u
        self.D = D
        self.gamma = gamma
        self.eta = eta
        self.E = E
        self.T0 = T0
        self.c_sat = c_sat
