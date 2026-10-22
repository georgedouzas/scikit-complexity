"""Energy balance climates: ice-albedo feedback, latitudinal heat transport, and Daisyworld."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import tanh

from ....core.base import Domain, Expression, state
from ....core.model import (
    Element,
    Field,
    LatticeSpace,
    Law,
    Observable,
    Rate,
    Symbols,
    System,
    gradient,
    laplacian,
    where,
)
from ....core.runtime import register_model


def _daisyworld_albedo(s: Any) -> Expression:  # noqa: ANN401
    """Return the planetary albedo, the area average of the albedos of bare ground and of both daisies."""
    bare = s.p - s.white - s.black
    return bare * s.albedo_ground + s.white * s.albedo_white + s.black * s.albedo_black


def _daisyworld_emission(s: Any) -> Expression:  # noqa: ANN401
    """Return the fourth power of the effective temperature of the planet."""
    return s.S * s.L * (1 - _daisyworld_albedo(s)) / s.sigma


def _daisyworld_growth(s: Any, albedo: Expression) -> Expression:  # noqa: ANN401
    """Return the growth rate of daisies of an albedo, parabolic in their local temperature and zero outside it."""
    local = (s.q * (_daisyworld_albedo(s) - albedo) + _daisyworld_emission(s)) ** (1 / 4)
    growth = 1 - s.k * (s.optimum - local) ** 2
    return where(growth > 0, growth, 0)


def _daisyworld_laws(s: Symbols) -> list[Law]:
    """Return the growth and death of white and black daisies on the bare ground."""
    bare = s.p - s.white - s.black
    return [
        Rate(s.white, s.white * (bare * _daisyworld_growth(s, s.albedo_white) - s.gamma)),
        Rate(s.black, s.black * (bare * _daisyworld_growth(s, s.albedo_black) - s.gamma)),
    ]


def _daisyworld_temperature(s: Any) -> Expression:  # noqa: ANN401
    """Return the effective temperature of Daisyworld."""
    return _daisyworld_emission(s.planet) ** (1 / 4)


def _budyko_sellers_laws(s: Symbols) -> list[Law]:
    """Return the absorbed insolation, the linearized outgoing radiation, and the meridional diffusion of heat."""
    legendre = (3 * s.x**2 - 1) / 2
    insolation = s.Q * (1 + s.s2 * legendre)
    free = s.albedo_free + s.albedo_slope * legendre
    albedo = (s.albedo_ice + free) / 2 - (s.albedo_ice - free) / 2 * tanh((s.T - s.T_ice) / s.width)
    transport = s.D * ((1 - s.x**2) * laplacian(s.T) - 2 * s.x * gradient(s.T, 1)[0])
    return [Rate(s.T, (insolation * (1 - albedo) - (s.A + s.B * s.T) + transport) / s.C)]


IceAlbedoClimate = Element.define(
    'IceAlbedoClimate',
    states={'T': state(default=290.0, domain=Domain.POSITIVE)},
    parameters={
        's': 1.0,
        'epsilon': 0.65,
        'sigma': 1.6e-10,
        'albedo': 0.5,
        'contrast': 0.2,
        'T_ice': 263.0,
        'width': 4.0,
    },
    laws=lambda s: [
        Rate(s.T, s.s * (1 - (s.albedo - s.contrast * tanh((s.T - s.T_ice) / s.width))) - s.sigma * s.epsilon * s.T**4),
    ],
)

LatitudeBand = Element.define(
    'LatitudeBand',
    states={'T': state(default=15.0, domain=Domain.REAL)},
    parameters={
        'x': 0.5,
        'Q': 334.4,
        's2': -0.482,
        'A': 203.3,
        'B': 2.09,
        'D': 0.649,
        'C': 10.0,
        'albedo_ice': 0.62,
        'albedo_free': 0.319,
        'albedo_slope': 0.202,
        'T_ice': -10.0,
        'width': 1.0,
    },
    laws=_budyko_sellers_laws,
)

Daisyworld = Element.define(
    'Daisyworld',
    states={
        'white': state(default=0.2, domain=Domain.NONNEGATIVE),
        'black': state(default=0.2, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'L': 1.0,
        'S': 917.0,
        'sigma': 5.67e-8,
        'q': 2.06e9,
        'p': 1.0,
        'gamma': 0.3,
        'k': 0.003265,
        'optimum': 295.5,
        'albedo_ground': 0.5,
        'albedo_white': 0.75,
        'albedo_black': 0.25,
    },
    laws=_daisyworld_laws,
)


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
    climate = IceAlbedoClimate(
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
    return System(components=[climate])


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
        >>> budyko_sellers(10).elements[0].size
        10
    """
    x = (np.arange(n_bands) + 0.5) / n_bands
    bands = Field(
        'bands',
        LatitudeBand,
        LatticeSpace(n_bands, spacing=1 / n_bands),
        boundary={'T': 'zero-flux'},
        values={'T': np.full(n_bands, float(T)), 'x': x, 'Q': Q, 'A': A, 'B': B, 'D': D, 'C': C},
    )
    return System(components=[bands])


@register_model(
    field='earth/climate',
    tags=('ode', 'regulation', 'Gaia', 'vegetation'),
    references=('Watson and Lovelock 1983', 'Jensen 2023'),
    defaults={'until': 100.0, 'dt': 0.5},
    observables=lambda system: [Observable('temperature', _daisyworld_temperature)],
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
        components=[Daisyworld('planet', white=white, black=black, L=L, gamma=gamma)],
    )
