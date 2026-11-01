"""Global climate of a zero-dimensional energy balance, bistable through the ice-albedo feedback."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, tanh
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


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
