"""Site of a quasi-geostrophic ocean basin, the relative potential vorticity of the water column at one position."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class QuasiGeostrophicSiteElement(Element):
    """Site of a quasi-geostrophic ocean basin, the relative potential vorticity of the water column at one position.

    The site holds the relative potential vorticity `zeta = laplacian(psi) - psi / Rd**2`, the potential vorticity
    without the planetary part `beta y`, with the planetary vorticity gradient `beta`, the deformation radius `Rd`, the
    viscosity `nu`, and the curl of the wind stress `curl` at the site.

    Args:
        label:
            The label of the site or the basin.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        zeta:
            The initial relative potential vorticity.

        beta:
            The gradient of the planetary vorticity.

        Rd:
            The deformation radius.

        nu:
            The viscosity.

        curl:
            The curl of the wind stress.

    Examples:
        >>> from skcomplex.models.physics.oceanography import QuasiGeostrophicSiteElement
        >>> QuasiGeostrophicSiteElement('site').beta
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        zeta: Annotated[float, StateSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        Rd: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        nu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1e-4,
        curl: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.zeta = zeta
        self.beta = beta
        self.Rd = Rd
        self.nu = nu
        self.curl = curl
