"""Latitude band of the Budyko-Sellers energy balance, heated by the sun and exchanging heat with its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, tanh

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import gradient, laplacian
from ....specs import ParameterSpec, StateSpec, VisualSpec


class LatitudeBandElement(Element):
    """Latitude band of the Budyko-Sellers energy balance, heated by the sun and exchanging heat with its neighbours.

    The temperature of the band at `x = sin(latitude)` follows `C dT/dt = Q s(x) (1 - a(x, T)) - (A + B T) +
    D d/dx((1 - x^2) dT/dx)`, with the annual insolation `s(x) = 1 + s2 P2(x)`, where `P2(x) = (3x^2 - 1) / 2`. The
    albedo is `albedo_free + albedo_slope P2(x)` without ice and rises to `albedo_ice` where the temperature falls below
    `T_ice`.

    Args:
        label:
            The label of the band or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature in °C.

        x:
            The sine of the latitude of the band.

        Q:
            The mean insolation in W m⁻².

        s2:
            The coefficient of the second Legendre polynomial of the insolation.

        A:
            The outgoing radiation at 0 °C in W m⁻².

        B:
            The increase of outgoing radiation per degree in W m⁻² K⁻¹.

        D:
            The coefficient of meridional heat transport in W m⁻² K⁻¹.

        C:
            The heat capacity in W yr m⁻² K⁻¹, so time is in years.

        albedo_ice:
            The albedo of an icy band.

        albedo_free:
            The mean albedo of an ice free band.

        albedo_slope:
            The coefficient of the second Legendre polynomial of the albedo of an ice free band.

        T_ice:
            The temperature in °C below which the band is icy.

        width:
            The width of the transition to ice in kelvin.

    Examples:
        >>> from skcomplex.models.earth.climate import LatitudeBandElement
        >>> LatitudeBandElement('band').albedo_ice
        0.62
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 15.0,
        x: Annotated[float, ParameterSpec()] = 0.5,
        Q: Annotated[float, ParameterSpec()] = 334.4,
        s2: Annotated[float, ParameterSpec()] = -0.482,
        A: Annotated[float, ParameterSpec()] = 203.3,
        B: Annotated[float, ParameterSpec()] = 2.09,
        D: Annotated[float, ParameterSpec()] = 0.649,
        C: Annotated[float, ParameterSpec()] = 10.0,
        albedo_ice: Annotated[float, ParameterSpec()] = 0.62,
        albedo_free: Annotated[float, ParameterSpec()] = 0.319,
        albedo_slope: Annotated[float, ParameterSpec()] = 0.202,
        T_ice: Annotated[float, ParameterSpec()] = -10.0,
        width: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.x = x
        self.Q = Q
        self.s2 = s2
        self.A = A
        self.B = B
        self.D = D
        self.C = C
        self.albedo_ice = albedo_ice
        self.albedo_free = albedo_free
        self.albedo_slope = albedo_slope
        self.T_ice = T_ice
        self.width = width

    def _get_albedo(self, s: SimpleNamespace, legendre: Expression) -> Expression:
        """Return the albedo of the band, which rises smoothly from its ice free value to that of ice as it cools.

        Args:
            s:
                The namespace of the symbols the laws can use.

            legendre:
                The second Legendre polynomial of the sine of the latitude.

        Returns:
            albedo:
                The albedo of the band.
        """
        free = s.albedo_free + s.albedo_slope * legendre
        return (s.albedo_ice + free) / 2 - (s.albedo_ice - free) / 2 * tanh((s.T - s.T_ice) / s.width)

    def _get_transport(self, s: SimpleNamespace) -> Expression:
        """Return the meridional diffusion of heat on the sphere, in the sine of latitude.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            transport:
                The heat the band gains from its neighbours.
        """
        return s.D * ((1 - s.x**2) * laplacian(s.T) - 2 * s.x * gradient(s.T, 1)[0])

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the absorbed insolation, the linearized outgoing radiation, and the meridional diffusion of heat.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the temperature of the band.
        """
        legendre = (3 * s.x**2 - 1) / 2
        insolation = s.Q * (1 + s.s2 * legendre)
        albedo = self._get_albedo(s, legendre)
        transport = self._get_transport(s)
        return [RateLaw(s.T, (insolation * (1 - albedo) - (s.A + s.B * s.T) + transport) / s.C)]
