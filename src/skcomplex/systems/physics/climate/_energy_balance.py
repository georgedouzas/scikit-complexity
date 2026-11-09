"""Energy balance climates.

The module covers ice-albedo feedback, latitudinal heat transport, and Daisyworld.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import gradient, laplacian, where
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class LatitudeBandElement(Element):
    """Latitude band of the Budyko-Sellers energy balance.

    The band is heated by the sun and exchanges heat with its neighbours.

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
        >>> from skcomplex.systems.physics.climate import LatitudeBandElement
        >>> LatitudeBandElement('band').albedo_ice
        0.62
    """

    law_classes = (RateLaw,)

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
        """Return the albedo of the band.

        The albedo rises smoothly from its ice free value to that of ice as the band cools.

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
        """Return the absorbed insolation, the outgoing radiation, and the heat diffusion.

        The outgoing radiation is linearized, and the diffusion of heat is meridional.

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


def _daisyworld_temperature(s: SimpleNamespace) -> Expression:
    """Return the effective temperature of Daisyworld."""
    return DaisyworldSystem._daisyworld_emission(s.planet) ** (1 / 4)


class EnergyBalanceSystem(DynamicalSystem):
    """The zero-dimensional energy balance of the Earth.

    The balance is bistable through the ice-albedo feedback.

    In units of the mean insolation, the global temperature follows `dT/dt = s (1 - a(T)) - sigma epsilon T^4`, with the
    albedo `a(T) = albedo - contrast tanh((T - T_ice) / width)`, high on an icy planet and low on a warm one. For the
    default parameters a warm climate near 286 K and a frozen climate near 232 K are both stable. The system has the
    temperature `earth.T`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import EnergyBalanceSystem
        >>> EnergyBalanceSystem(T=230.0).prepare().system_.elements[0].get_value('T')
        230.0
    """

    field = 'physics/climate'
    topics = ('ode', 'bistability', 'ice-albedo feedback', 'energy balance')
    references = ('Budyko 1969', 'Sellers 1969', 'Datseris and Parlitz 2022', 'Ghil and Lucarini 2020')
    until = 500.0
    dt = 5.0

    def __init__(
        self,
        T: float = 290.0,
        *,
        s: float = 1.0,
        epsilon: float = 0.65,
        sigma: float = 1.6e-10,
        albedo: float = 0.5,
        contrast: float = 0.2,
        T_ice: float = 263.0,
        width: float = 4.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.T = T
        self.s = s
        self.epsilon = epsilon
        self.sigma = sigma
        self.albedo = albedo
        self.contrast = contrast
        self.T_ice = T_ice
        self.width = width

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'earth'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'T': DeclaredVariableSpec(self.T, StateSpec(domain=RealSet.open(0, oo))),
            's': DeclaredVariableSpec(self.s, ParameterSpec()),
            'epsilon': DeclaredVariableSpec(self.epsilon, ParameterSpec()),
            'sigma': DeclaredVariableSpec(self.sigma, ParameterSpec()),
            'albedo': DeclaredVariableSpec(self.albedo, ParameterSpec()),
            'contrast': DeclaredVariableSpec(self.contrast, ParameterSpec()),
            'T_ice': DeclaredVariableSpec(self.T_ice, ParameterSpec()),
            'width': DeclaredVariableSpec(self.width, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the absorbed insolation and the outgoing radiation."""
        return {
            'T': lambda s: (
                s.target.s
                * (1 - (s.target.albedo - s.target.contrast * tanh((s.target.T - s.target.T_ice) / s.target.width)))
                - s.target.sigma * s.target.epsilon * s.target.T**4
            ),
        }


class BudykoSellersSystem(BaseSystem):
    """The Budyko-Sellers energy balance of one hemisphere.

    The balance is resolved in the sine of latitude.

    The temperature of a band at `x = sin(latitude)` follows `C dT/dt = Q s(x) (1 - a(x, T)) - (A + B T) + D d/dx((1 -
    x^2) dT/dx)`, with the annual insolation `s(x) = 1 - 0.482 P2(x)`, where `P2(x) = (3x^2 - 1) / 2`. The albedo is
    `0.319 + 0.202 P2(x)` without ice and rises to 0.62 where the temperature falls below -10 °C. Bands are centred at
    `x = (i + 1/2) / n_bands`, with no heat flux across the equator or the pole. Present insolation keeps both a climate
    with polar ice caps and a snowball Earth. The system has the temperatures `bands.T` from the equator to the pole.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import BudykoSellersSystem
        >>> BudykoSellersSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/climate'
    topics = ('field', 'pde', 'energy balance', 'ice-albedo feedback', 'snowball')
    references = ('Budyko 1969', 'Sellers 1969', 'North 1975', 'Ghil and Lucarini 2020')
    until = 100.0
    dt = 2.0

    def __init__(
        self,
        n_bands: int = 30,
        T: float = 15.0,
        *,
        Q: float = 334.4,
        A: float = 203.3,
        B: float = 2.09,
        D: float = 0.649,
        C: float = 10.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_bands = n_bands
        self.T = T
        self.Q = Q
        self.A = A
        self.B = B
        self.D = D
        self.C = C

    def _build_system(self) -> CompositeSystem:
        """Return the Budyko-Sellers energy balance of one hemisphere.

        The balance is resolved in the sine of latitude.
        """
        x = (np.arange(self.n_bands) + 0.5) / self.n_bands
        bands = FieldElement(
            'bands',
            element=LatitudeBandElement,
            space=LatticeSpace(self.n_bands, spacing=1 / self.n_bands),
            boundary={'T': 'zero-flux'},
            values={
                'T': np.full(self.n_bands, float(self.T)),
                'x': x,
                'Q': self.Q,
                'A': self.A,
                'B': self.B,
                'D': self.D,
                'C': self.C,
            },
        )
        return CompositeSystem(self.label, elements=[bands])


class DaisyworldSystem(DynamicalSystem):
    """Daisyworld.

    White and black daisies regulate the temperature of their planet through its albedo.

    The areas covered by white and black daisies grow on the bare ground at a rate parabolic in their local temperature,
    with an optimum at 22.5 °C, and die at the rate `gamma`. The effective temperature follows the planetary albedo,
    `sigma T^4 = S L (1 - A)`, and is recorded as the observable `temperature`. The system has the areas `planet.white`
    and `planet.black` and the observable `temperature`.

    Args:
        white:
            The initial area fraction of white daisies.

        black:
            The initial area fraction of black daisies.

        L:
            The luminosity of the sun relative to its present value.

        gamma:
            The death rate of daisies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import DaisyworldSystem
        >>> DaisyworldSystem(L=1.2).prepare().system_.elements[0].get_value('L')
        1.2
    """

    field = 'physics/climate'
    topics = ('ode', 'regulation', 'Gaia', 'vegetation')
    references = ('Watson and Lovelock 1983', 'Jensen 2023')
    until = 100.0
    dt = 0.5
    observables = (ObservableSpec('temperature', _daisyworld_temperature),)

    def __init__(
        self,
        white: float = 0.2,
        black: float = 0.2,
        *,
        L: float = 1.0,
        gamma: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.white = white
        self.black = black
        self.L = L
        self.gamma = gamma

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'planet'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'white': DeclaredVariableSpec(self.white, StateSpec(domain=RealSet.closed_open(0, oo))),
            'black': DeclaredVariableSpec(self.black, StateSpec(domain=RealSet.closed_open(0, oo))),
            'L': DeclaredVariableSpec(self.L, ParameterSpec()),
            'S': DeclaredVariableSpec(917.0, ParameterSpec()),
            'sigma': DeclaredVariableSpec(5.67e-08, ParameterSpec()),
            'q': DeclaredVariableSpec(2060000000.0, ParameterSpec()),
            'p': DeclaredVariableSpec(1.0, ParameterSpec()),
            'gamma': DeclaredVariableSpec(self.gamma, ParameterSpec()),
            'k': DeclaredVariableSpec(0.003265, ParameterSpec()),
            'optimum': DeclaredVariableSpec(295.5, ParameterSpec()),
            'albedo_ground': DeclaredVariableSpec(0.5, ParameterSpec()),
            'albedo_white': DeclaredVariableSpec(0.75, ParameterSpec()),
            'albedo_black': DeclaredVariableSpec(0.25, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the growth and death of white and black daisies on the bare ground."""
        return {
            'white': lambda s: (
                s.target.white * (self._bare(s.target) * self._growth(s.target, s.target.albedo_white) - s.target.gamma)
            ),
            'black': lambda s: (
                s.target.black * (self._bare(s.target) * self._growth(s.target, s.target.albedo_black) - s.target.gamma)
            ),
        }

    @staticmethod
    def _bare(planet: SimpleNamespace) -> Expression:
        """Return the area fraction of bare fertile ground."""
        return planet.p - planet.white - planet.black

    @staticmethod
    def _growth(planet: SimpleNamespace, albedo: Expression) -> Expression:
        """Return the growth rate of daisies of an albedo.

        The rate is parabolic in their local temperature and zero outside it.
        """
        local = (
            planet.q * (DaisyworldSystem._daisyworld_albedo(planet) - albedo)
            + DaisyworldSystem._daisyworld_emission(planet)
        ) ** (1 / 4)
        growth = 1 - planet.k * (planet.optimum - local) ** 2
        return where(growth > 0, growth, 0)

    @staticmethod
    def _daisyworld_albedo(planet: SimpleNamespace) -> Expression:
        """Return the planetary albedo.

        The albedo is the area average of the albedos of bare ground and of both daisies.
        """
        bare = planet.p - planet.white - planet.black
        return bare * planet.albedo_ground + planet.white * planet.albedo_white + planet.black * planet.albedo_black

    @staticmethod
    def _daisyworld_emission(planet: SimpleNamespace) -> Expression:
        """Return the fourth power of the effective temperature of the planet."""
        return planet.S * planet.L * (1 - DaisyworldSystem._daisyworld_albedo(planet)) / planet.sigma
