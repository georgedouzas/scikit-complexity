"""Energy balance climates.

The module covers ice-albedo feedback, latitudinal heat transport, and Daisyworld.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import gradient, laplacian, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class LatitudeBandElement(Element):
    """Latitude bands of the Budyko-Sellers energy balance, their temperatures and positions.

    Every band sits at `x = sin(latitude)` and holds its temperature `T` in °C.

    Args:
        label:
            The label of the bands.

        T:
            The initial temperature in °C, one value or one per band.

        x:
            The sine of the latitude, one value or one per band.

        space:
            The lattice of the bands, or `None` for one band.

    Examples:
        >>> from skcomplex.systems.physics.climate import LatitudeBandElement
        >>> LatitudeBandElement('band', x=0.25).x
        0.25
    """

    def __init__(
        self,
        label: str,
        *,
        T: object = 15.0,
        x: object = 0.5,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.T = T
        self.x = x


class LatitudeEnergyBalanceInteraction(Interaction):
    """Energy balance of latitude bands heated by the sun and exchanging heat with their neighbours.

    The temperature of a band at `x = sin(latitude)` follows `C dT/dt = Q s(x) (1 - a(x, T)) - (A + B T) + D d/dx((1 -
    x^2) dT/dx)`, with the annual insolation `s(x) = 1 + s2 P2(x)`, where `P2(x) = (3x^2 - 1) / 2`. The albedo is
    `albedo_free + albedo_slope P2(x)` without ice and rises to `albedo_ice` where the temperature falls below `T_ice`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bands.

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
        >>> from skcomplex.systems.physics.climate import LatitudeEnergyBalanceInteraction
        >>> LatitudeEnergyBalanceInteraction('balance', target='bands').albedo_ice
        0.62
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        Q: float = 334.4,
        s2: float = -0.482,
        A: float = 203.3,
        B: float = 2.09,
        D: float = 0.649,
        C: float = 10.0,
        albedo_ice: float = 0.62,
        albedo_free: float = 0.319,
        albedo_slope: float = 0.202,
        T_ice: float = -10.0,
        width: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
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

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the absorbed insolation, the outgoing radiation, and the meridional diffusion of heat.

        Args:
            context:
                The namespace of the bands, the time `t`, and the parameters.
        """
        super()._prepare(context)
        T, x = context.target.T, context.target.x
        legendre = (3 * x**2 - 1) / 2
        insolation = context.Q * (1 + context.s2 * legendre)
        free = context.albedo_free + context.albedo_slope * legendre
        ice = context.albedo_ice
        albedo = (ice + free) / 2 - (ice - free) / 2 * tanh((T - context.T_ice) / context.width)
        transport = context.D * ((1 - x**2) * laplacian(T) - 2 * x * gradient(T, 1)[0])
        balance = (insolation * (1 - albedo) - (context.A + context.B * T) + transport) / context.C
        self.laws_ = [RateLaw(T.diff(context.t), balance)]


class IceAlbedoBalanceInteraction(Interaction):
    """Global energy balance of a planet with the ice-albedo feedback.

    In units of the mean insolation, the temperature `T` of the target follows `T' = s (1 - a(T)) - sigma epsilon T^4`,
    with the albedo `a(T) = albedo - contrast tanh((T - T_ice) / width)`, high on an icy planet and low on a warm one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the planet.

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
        >>> from skcomplex.systems.physics.climate import IceAlbedoBalanceInteraction
        >>> IceAlbedoBalanceInteraction('balance', target='earth', epsilon=0.6).epsilon
        0.6
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        s: float = 1.0,
        epsilon: float = 0.65,
        sigma: float = 1.6e-10,
        albedo: float = 0.5,
        contrast: float = 0.2,
        T_ice: float = 263.0,
        width: float = 4.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.s = s
        self.epsilon = epsilon
        self.sigma = sigma
        self.albedo = albedo
        self.contrast = contrast
        self.T_ice = T_ice
        self.width = width

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the temperature, the absorbed insolation minus the outgoing radiation.

        Args:
            context:
                The namespace of the planet, the time `t`, and the parameters.
        """
        super()._prepare(context)
        T = context.target.T
        albedo = context.albedo - context.contrast * tanh((T - context.T_ice) / context.width)
        radiation = context.sigma * context.epsilon * T**4
        self.laws_ = [RateLaw(T.diff(context.t), context.s * (1 - albedo) - radiation)]


class EnergyBalanceSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import EnergyBalanceSystem
        >>> trajectory = EnergyBalanceSystem(T=230.0).prepare().simulate(until=5.0)
        >>> float(trajectory.get_values('earth.T')[0])
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.T = T
        self.s = s
        self.epsilon = epsilon
        self.sigma = sigma
        self.albedo = albedo
        self.contrast = contrast
        self.T_ice = T_ice
        self.width = width

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the global temperature and its energy balance.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        balance = IceAlbedoBalanceInteraction(
            'balance',
            target='earth',
            s=self.s,
            epsilon=self.epsilon,
            sigma=self.sigma,
            albedo=self.albedo,
            contrast=self.contrast,
            T_ice=self.T_ice,
            width=self.width,
        )
        self.system_ = System(self.label, elements=[Element('earth', values={'T': self.T})], interactions=[balance])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import BudykoSellersSystem
        >>> trajectory = BudykoSellersSystem(10).prepare().simulate(until=2.0)
        >>> trajectory.get_values('bands.T').shape
        (2, 10)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_bands = n_bands
        self.T = T
        self.Q = Q
        self.A = A
        self.B = B
        self.D = D
        self.C = C

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the latitude bands of one hemisphere and their energy balance.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        bands = LatitudeBandElement(
            'bands',
            T=np.full(self.n_bands, float(self.T)),
            x=(np.arange(self.n_bands) + 0.5) / self.n_bands,
            space=LatticeSpace(self.n_bands, spacing=1 / self.n_bands, boundary='reflecting'),
        )
        balance = LatitudeEnergyBalanceInteraction(
            'balance',
            target='bands',
            Q=self.Q,
            A=self.A,
            B=self.B,
            D=self.D,
            C=self.C,
        )
        self.system_ = System(self.label, elements=[bands], interactions=[balance])


def _get_daisyworld_temperature(s: SimpleNamespace) -> Expression:
    """Return the effective temperature of Daisyworld."""
    return _daisyworld_emission(s.planet) ** (1 / 4)


def _daisyworld_emission(planet: SimpleNamespace) -> Expression:
    """Return the fourth power of the effective temperature of the planet."""
    return planet.S * planet.L * (1 - DaisyworldSystem._get_daisyworld_albedo(planet)) / planet.sigma


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
        >>> DaisyworldSystem(L=1.2).prepare().system_.elements[0].values['L']
        1.2
    """

    field = 'physics/climate'
    topics = ('ode', 'regulation', 'Gaia', 'vegetation')
    references = ('Watson and Lovelock 1983', 'Jensen 2023')
    until = 100.0
    dt = 0.5
    observables = (Observable('temperature', _get_daisyworld_temperature),)

    def __init__(
        self,
        white: float = 0.2,
        black: float = 0.2,
        *,
        L: float = 1.0,
        gamma: float = 0.3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.white = white
        self.black = black
        self.L = L
        self.gamma = gamma

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'planet'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'white': self.white,
            'black': self.black,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the states and parameters that have them, by name."""
        return {
            'white': RealSet.closed_open(0, oo),
            'black': RealSet.closed_open(0, oo),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'L': self.L,
            'S': 917.0,
            'sigma': 5.67e-08,
            'q': 2060000000.0,
            'p': 1.0,
            'gamma': self.gamma,
            'k': 0.003265,
            'optimum': 295.5,
            'albedo_ground': 0.5,
            'albedo_white': 0.75,
            'albedo_black': 0.25,
        }

    def _get_equations(self) -> Equations:
        """Return the growth and death of white and black daisies on the bare ground."""
        return {
            'white': lambda s: (
                s.target.white
                * (self._get_bare(s.target) * self._get_growth(s.target, s.target.albedo_white) - s.target.gamma)
            ),
            'black': lambda s: (
                s.target.black
                * (self._get_bare(s.target) * self._get_growth(s.target, s.target.albedo_black) - s.target.gamma)
            ),
        }

    @staticmethod
    def _get_bare(planet: SimpleNamespace) -> Expression:
        """Return the area fraction of bare fertile ground."""
        return planet.p - planet.white - planet.black

    @staticmethod
    def _get_growth(planet: SimpleNamespace, albedo: Expression) -> Expression:
        """Return the growth rate of daisies of an albedo.

        The rate is parabolic in their local temperature and zero outside it.
        """
        local = (
            planet.q * (DaisyworldSystem._get_daisyworld_albedo(planet) - albedo) + _daisyworld_emission(planet)
        ) ** (1 / 4)
        growth = 1 - planet.k * (planet.optimum - local) ** 2
        return where(growth > 0, growth, 0)

    @staticmethod
    def _get_daisyworld_albedo(planet: SimpleNamespace) -> Expression:
        """Return the planetary albedo.

        The albedo is the area average of the albedos of bare ground and of both daisies.
        """
        bare = planet.p - planet.white - planet.black
        return bare * planet.albedo_ground + planet.white * planet.albedo_white + planet.black * planet.albedo_black
