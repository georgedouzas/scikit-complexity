"""Define the site of an active nematic gel, a tensor carried by the Stokes flow of its own stresses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import inverse_laplacian, leray_projection, tensor_divergence
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..common.fields import corotational
from ._landau_de_gennes_energy import LandauDeGennesEnergyInteraction


class NematicGelSiteElement(Element):
    """Site of an active nematic gel, a symmetric traceless tensor `Q` carried by an overdamped Stokes flow.

    The tensor follows `Q_t = -corotational(Q, u, xi) + H / gamma`, with the molecular field `H` of the Landau-de Gennes
    energy. The velocity solves `eta laplacian(u) - grad(P) + div(sigma) = 0`, `div(u) = 0`, with the stress
    `sigma = -xi H + Q H - H Q - zeta Q`. An activity `zeta > 0` is extensile and an activity `zeta < 0` contractile.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        Q:
            The initial tensor, its independent components `Q_xx` and `Q_xy`.

        a:
            The coefficient of `tr(Q^2) / 2` of the energy, negative in the ordered phase.

        c:
            The coefficient of `tr(Q^2)^2 / 4` of the energy.

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
        >>> from skcomplex.models.physics.active_matter import NematicGelSiteElement
        >>> NematicGelSiteElement('site').zeta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Q: Annotated[npt.ArrayLike, StateSpec(tensor='traceless')] = 0.0,
        a: Annotated[float, ParameterSpec()] = -1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        xi: Annotated[float, ParameterSpec()] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Q = Q
        self.a = a
        self.c = c
        self.K = K
        self.xi = xi
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta

    @staticmethod
    def _velocity(s: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity of the Stokes flow of the elastic and active stresses at a site."""
        field = LandauDeGennesEnergyInteraction._molecular_field(s.Q, s.a, s.c, s.K)
        stress = -s.xi * field + s.Q * field - field * s.Q - s.zeta * s.Q
        force = tensor_divergence(stress)
        return leray_projection(force.apply_map(lambda item: inverse_laplacian(-item / s.eta)))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the tensor, its transport in the flow and its relaxation.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of `Q`.
        """
        s = namespace
        field = LandauDeGennesEnergyInteraction._molecular_field(s.Q, s.a, s.c, s.K)
        return [RateLaw(s.Q, -corotational(s.Q, self._velocity(s), s.xi) + field / s.gamma)]
