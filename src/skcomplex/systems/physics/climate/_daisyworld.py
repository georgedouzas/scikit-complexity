"""Planet of Daisyworld, whose white and black daisies regulate its temperature through its albedo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


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
        >>> from skcomplex.systems.physics.climate import DaisyworldElement
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
        local = (
            s.q * (DaisyworldElement._daisyworld_albedo(s) - albedo) + DaisyworldElement._daisyworld_emission(s)
        ) ** (1 / 4)
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

    @staticmethod
    def _daisyworld_albedo(s: SimpleNamespace) -> Expression:
        """Return the planetary albedo, the area average of the albedos of bare ground and of both daisies."""
        bare = s.p - s.white - s.black
        return bare * s.albedo_ground + s.white * s.albedo_white + s.black * s.albedo_black

    @staticmethod
    def _daisyworld_emission(s: SimpleNamespace) -> Expression:
        """Return the fourth power of the effective temperature of the planet."""
        return s.S * s.L * (1 - DaisyworldElement._daisyworld_albedo(s)) / s.sigma
