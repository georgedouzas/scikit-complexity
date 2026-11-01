"""Energy balance climates: ice-albedo feedback, latitudinal heat transport, and Daisyworld."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import FieldElement
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._daisyworld import DaisyworldElement
from ._ice_albedo_climate import IceAlbedoClimateElement
from ._latitude_band import LatitudeBandElement


def _daisyworld_temperature(s: SimpleNamespace) -> Expression:
    """Return the effective temperature of Daisyworld."""
    return DaisyworldElement._daisyworld_emission(s.planet) ** (1 / 4)


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
