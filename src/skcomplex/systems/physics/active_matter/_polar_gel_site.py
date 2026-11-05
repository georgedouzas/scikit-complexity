"""Define the site of an active polar gel, a polarisation carried by the Stokes flow of its own stresses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import inverse_laplacian, laplacian, leray_projection, tensor_divergence
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.fields import corotational


class PolarGelSiteElement(Element):
    """Site of an active polar gel, a polarisation `p` carried by an overdamped Stokes flow.

    The polarisation follows `p_t = -corotational(p, u, xi) + h / gamma`, with the molecular field
    `h = -a p - c |p|^2 p + K laplacian(p)`. The velocity solves `eta laplacian(u) - grad(P) + div(sigma) = 0`,
    `div(u) = 0`, with the stress `sigma = -xi (p h + h p) / 2 + (p h - h p) / 2 - zeta p p`, as in Kruse et al.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        p:
            The initial polarisation.

        a:
            The coefficient of `|p|^2 / 2` of the energy, negative in the polar phase.

        c:
            The coefficient of `|p|^4 / 4` of the energy.

        K:
            The elastic constant.

        xi:
            The flow-alignment parameter.

        gamma:
            The rotational viscosity.

        eta:
            The viscosity of the flow.

        zeta:
            The activity.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PolarGelSiteElement
        >>> PolarGelSiteElement('site').xi
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        xi: Annotated[float, ParameterSpec()] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec()] = -0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.a = a
        self.c = c
        self.K = K
        self.xi = xi
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta

    @staticmethod
    def _molecular_field(s: SimpleNamespace) -> FreeModuleElement:
        """Return the molecular field of the polarisation at a site."""
        return -s.a * s.p - s.c * s.p.dot_product(s.p) * s.p + s.K * s.p.apply_map(laplacian)

    @staticmethod
    def _velocity(s: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity of the Stokes flow of the elastic and active stresses at a site."""
        field = PolarGelSiteElement._molecular_field(s)
        outer = s.p.outer_product(field)
        stress = -s.xi * (outer + outer.transpose()) / 2 + (outer - outer.transpose()) / 2
        stress -= s.zeta * s.p.outer_product(s.p)
        force = tensor_divergence(stress)
        return leray_projection(force.apply_map(lambda item: inverse_laplacian(-item / s.eta)))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the polarisation, its transport in the flow and its relaxation.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `p`.
        """
        s = namespace
        return [RateLaw(s.p, -corotational(s.p, self._velocity(s), s.xi) + self._molecular_field(s) / s.gamma)]
