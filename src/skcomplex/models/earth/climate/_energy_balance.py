"""Energy balance climates: ice-albedo feedback, latitudinal heat transport, and Daisyworld."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import gradient, laplacian, where
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _daisyworld_albedo(s: SimpleNamespace) -> Expression:
    """Return the planetary albedo, the area average of the albedos of bare ground and of both daisies."""
    bare = s.p - s.white - s.black
    return bare * s.albedo_ground + s.white * s.albedo_white + s.black * s.albedo_black


def _daisyworld_emission(s: SimpleNamespace) -> Expression:
    """Return the fourth power of the effective temperature of the planet."""
    return s.S * s.L * (1 - _daisyworld_albedo(s)) / s.sigma


def _daisyworld_temperature(s: SimpleNamespace) -> Expression:
    """Return the effective temperature of Daisyworld."""
    return _daisyworld_emission(s.planet) ** (1 / 4)


class IceAlbedoClimateElement(Element):
    """Global climate of a zero-dimensional energy balance, bistable through the ice-albedo feedback.

    In units of the mean insolation, the global temperature follows `dT/dt = s (1 - a(T)) - sigma epsilon T^4`, with
    the albedo `a(T) = albedo - contrast tanh((T - T_ice) / width)`, high on an icy planet and low on a warm one.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial global mean temperature in kelvin.

        s:
            The insolation relative to the present one.

        epsilon:
            The effective emissivity, lowered by the greenhouse effect.

        sigma:
            The Stefan-Boltzmann constant divided by the mean insolation.

        albedo:
            The albedo at the middle of the transition from ice to no ice.

        contrast:
            Half the difference between the albedos of a frozen and of a warm planet.

        T_ice:
            The temperature of the middle of the transition.

        width:
            The width of the transition in kelvin.

    Examples:
        >>> from skcomplex.models.earth.climate import IceAlbedoClimateElement
        >>> IceAlbedoClimateElement('earth').T_ice
        263.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 290.0,
        s: Annotated[float, ParameterSpec()] = 1.0,
        epsilon: Annotated[float, ParameterSpec()] = 0.65,
        sigma: Annotated[float, ParameterSpec()] = 1.6e-10,
        albedo: Annotated[float, ParameterSpec()] = 0.5,
        contrast: Annotated[float, ParameterSpec()] = 0.2,
        T_ice: Annotated[float, ParameterSpec()] = 263.0,
        width: Annotated[float, ParameterSpec()] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.s = s
        self.epsilon = epsilon
        self.sigma = sigma
        self.albedo = albedo
        self.contrast = contrast
        self.T_ice = T_ice
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the absorbed insolation, lowered by the albedo of the ice, and the outgoing radiation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the temperature.
        """
        return [
            RateLaw(
                s.T,
                s.s * (1 - (s.albedo - s.contrast * tanh((s.T - s.T_ice) / s.width))) - s.sigma * s.epsilon * s.T**4,
            ),
        ]


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


class DaisyworldElement(Element):
    """Planet of Daisyworld, whose white and black daisies regulate its temperature through its albedo.

    The areas covered by white and black daisies grow on the bare ground at a rate parabolic in their local
    temperature, with an optimum at `optimum`, and die at the rate `gamma`. The effective temperature follows the
    planetary albedo, `sigma T^4 = S L (1 - A)`, and the local temperature of a daisy is warmer or cooler than it by
    `q` times the difference of their albedos.

    Args:
        label:
            The label of the planet.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        white:
            The initial area fraction of white daisies.

        black:
            The initial area fraction of black daisies.

        L:
            The luminosity of the sun relative to its present value.

        S:
            The flux of the present sun in W m⁻².

        sigma:
            The Stefan-Boltzmann constant in W m⁻² K⁻⁴.

        q:
            The transfer of heat between areas of different albedo, in K⁴.

        p:
            The fraction of the planet fertile for daisies.

        gamma:
            The death rate of daisies.

        k:
            The curvature of the parabola of growth in K⁻².

        optimum:
            The local temperature of the fastest growth in kelvin.

        albedo_ground:
            The albedo of the bare ground.

        albedo_white:
            The albedo of white daisies.

        albedo_black:
            The albedo of black daisies.

    Examples:
        >>> from skcomplex.models.earth.climate import DaisyworldElement
        >>> DaisyworldElement('planet').albedo_white
        0.75
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        white: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        black: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        L: Annotated[float, ParameterSpec()] = 1.0,
        S: Annotated[float, ParameterSpec()] = 917.0,
        sigma: Annotated[float, ParameterSpec()] = 5.67e-08,
        q: Annotated[float, ParameterSpec()] = 2060000000.0,
        p: Annotated[float, ParameterSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 0.3,
        k: Annotated[float, ParameterSpec()] = 0.003265,
        optimum: Annotated[float, ParameterSpec()] = 295.5,
        albedo_ground: Annotated[float, ParameterSpec()] = 0.5,
        albedo_white: Annotated[float, ParameterSpec()] = 0.75,
        albedo_black: Annotated[float, ParameterSpec()] = 0.25,
    ) -> None:
        super().__init__(label, visual=visual)
        self.white = white
        self.black = black
        self.L = L
        self.S = S
        self.sigma = sigma
        self.q = q
        self.p = p
        self.gamma = gamma
        self.k = k
        self.optimum = optimum
        self.albedo_ground = albedo_ground
        self.albedo_white = albedo_white
        self.albedo_black = albedo_black

    def _get_growth(self, s: SimpleNamespace, albedo: Expression) -> Expression:
        """Return the growth rate of daisies of an albedo, parabolic in their local temperature and zero outside it.

        Args:
            s:
                The namespace of the symbols the laws can use.

            albedo:
                The albedo of the daisies.

        Returns:
            growth:
                The growth rate of the daisies.
        """
        local = (s.q * (_daisyworld_albedo(s) - albedo) + _daisyworld_emission(s)) ** (1 / 4)
        growth = 1 - s.k * (s.optimum - local) ** 2
        return where(growth > 0, growth, 0)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and death of white and black daisies on the bare ground.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the areas of white and black daisies.
        """
        bare = s.p - s.white - s.black
        return [
            RateLaw(s.white, s.white * (bare * self._get_growth(s, s.albedo_white) - s.gamma)),
            RateLaw(s.black, s.black * (bare * self._get_growth(s, s.albedo_black) - s.gamma)),
        ]


@register_model(
    field='earth/climate',
    tags=('ode', 'bistability', 'ice-albedo feedback', 'energy balance'),
    references=('Budyko 1969', 'Sellers 1969', 'Datseris and Parlitz 2022', 'Ghil and Lucarini 2020'),
    defaults={'until': 500.0, 'dt': 5.0},
)
def energy_balance(
    T: float = 290.0,
    *,
    s: float = 1.0,
    epsilon: float = 0.65,
    sigma: float = 1.6e-10,
    albedo: float = 0.5,
    contrast: float = 0.2,
    T_ice: float = 263.0,
    width: float = 4.0,
) -> System:
    """Return the zero-dimensional energy balance of the Earth, bistable through the ice-albedo feedback.

    In units of the mean insolation, the global temperature follows `dT/dt = s (1 - a(T)) - sigma epsilon T^4`, with
    the albedo `a(T) = albedo - contrast tanh((T - T_ice) / width)`, high on an icy planet and low on a warm one. For
    the default parameters a warm climate near 286 K and a frozen climate near 232 K are both stable.

    Args:
        T:
            The initial global mean temperature in kelvin.

        s:
            The insolation relative to the present one.

        epsilon:
            The effective emissivity, lowered by the greenhouse effect.

        sigma:
            The Stefan-Boltzmann constant divided by the mean insolation.

        albedo:
            The albedo at the middle of the transition from ice to no ice.

        contrast:
            Half the difference between the albedos of a frozen and of a warm planet.

        T_ice:
            The temperature of the middle of the transition.

        width:
            The width of the transition in kelvin.

    Returns:
        system:
            The model, with the temperature `earth.T`.

    Examples:
        >>> from skcomplex.models.earth.climate import energy_balance
        >>> energy_balance(T=230.0).elements[0].T
        230.0
    """
    climate = IceAlbedoClimateElement(
        'earth',
        T=T,
        s=s,
        epsilon=epsilon,
        sigma=sigma,
        albedo=albedo,
        contrast=contrast,
        T_ice=T_ice,
        width=width,
    )
    return System(elements=[climate])


@register_model(
    field='earth/climate',
    tags=('field', 'pde', 'energy balance', 'ice-albedo feedback', 'snowball'),
    references=('Budyko 1969', 'Sellers 1969', 'North 1975', 'Ghil and Lucarini 2020'),
    defaults={'until': 100.0, 'dt': 2.0},
)
def budyko_sellers(
    n_bands: int = 30,
    T: float = 15.0,
    *,
    Q: float = 334.4,
    A: float = 203.3,
    B: float = 2.09,
    D: float = 0.649,
    C: float = 10.0,
) -> System:
    """Return the Budyko-Sellers energy balance of one hemisphere, resolved in the sine of latitude.

    The temperature of a band at `x = sin(latitude)` follows `C dT/dt = Q s(x) (1 - a(x, T)) - (A + B T) +
    D d/dx((1 - x^2) dT/dx)`, with the annual insolation `s(x) = 1 - 0.482 P2(x)`, where `P2(x) = (3x^2 - 1) / 2`.
    The albedo is `0.319 + 0.202 P2(x)` without ice and rises to 0.62 where the temperature falls below -10 °C.
    Bands are centred at `x = (i + 1/2) / n_bands`, with no heat flux across the equator or the pole. Present
    insolation keeps both a climate with polar ice caps and a snowball Earth.

    Args:
        n_bands:
            The number of latitude bands.

        T:
            The initial temperature of every band in °C.

        Q:
            The mean insolation in W m⁻².

        A:
            The outgoing radiation at 0 °C in W m⁻².

        B:
            The increase of outgoing radiation per degree in W m⁻² K⁻¹.

        D:
            The coefficient of meridional heat transport in W m⁻² K⁻¹.

        C:
            The heat capacity in W yr m⁻² K⁻¹, so time is in years.

    Returns:
        system:
            The model, with the temperatures `bands.T` from the equator to the pole.

    Examples:
        >>> from skcomplex.models.earth.climate import budyko_sellers
        >>> budyko_sellers(10).elements[0].check().prepare().size_
        10
    """
    x = (np.arange(n_bands) + 0.5) / n_bands
    bands = FieldElement(
        'bands',
        element=LatitudeBandElement,
        space=LatticeSpace(n_bands, spacing=1 / n_bands),
        boundary={'T': 'zero-flux'},
        values={'T': np.full(n_bands, float(T)), 'x': x, 'Q': Q, 'A': A, 'B': B, 'D': D, 'C': C},
    )
    return System(elements=[bands])


@register_model(
    field='earth/climate',
    tags=('ode', 'regulation', 'Gaia', 'vegetation'),
    references=('Watson and Lovelock 1983', 'Jensen 2023'),
    defaults={'until': 100.0, 'dt': 0.5},
    observables=lambda system: [ObservableSpec('temperature', _daisyworld_temperature)],
)
def daisyworld(
    white: float = 0.2,
    black: float = 0.2,
    *,
    L: float = 1.0,
    gamma: float = 0.3,
) -> System:
    """Return Daisyworld, where white and black daisies regulate the temperature of their planet through its albedo.

    The areas covered by white and black daisies grow on the bare ground at a rate parabolic in their local
    temperature, with an optimum at 22.5 °C, and die at the rate `gamma`. The effective temperature follows the
    planetary albedo, `sigma T^4 = S L (1 - A)`, and is recorded as the observable `temperature`.

    Args:
        white:
            The initial area fraction of white daisies.

        black:
            The initial area fraction of black daisies.

        L:
            The luminosity of the sun relative to its present value.

        gamma:
            The death rate of daisies.

    Returns:
        system:
            The model, with the areas `planet.white` and `planet.black` and the observable `temperature`.

    Examples:
        >>> from skcomplex.models.earth.climate import daisyworld
        >>> daisyworld(L=1.2).elements[0].L
        1.2
    """
    return System(
        elements=[DaisyworldElement('planet', white=white, black=black, L=L, gamma=gamma)],
    )
