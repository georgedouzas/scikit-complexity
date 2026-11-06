"""Satellite on a planar Kepler orbit whose radial thrust moves it to an orbit of a final energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import derivative, where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SatelliteElement(Element):
    """Satellite on a planar Kepler orbit whose radial thrust moves it to an orbit of a final energy."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[object, StateSpec(domain=RealSet.open(0, oo), order=2)] = 0.314,
        s_d1: float = 0.0,
        theta: Annotated[object, StateSpec()] = 0.0,
        Lz: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.7,
        epsilon: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        H_final: Annotated[object, ParameterSpec()] = -0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.s_d1 = s_d1
        self.theta = theta
        self.Lz = Lz
        self.epsilon = epsilon
        self.H_final = H_final

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the radial rate under thrust and the rate of the angle."""
        return [
            RateLaw(s.s, s.Lz**2 / s.s**3 - 1 / s.s**2 - s.epsilon * SatelliteElement._thrust_direction(s)),
            RateLaw(s.theta, s.Lz / s.s**2),
        ]

    @staticmethod
    def _satellite_energy(s: SimpleNamespace) -> Expression:
        """Return the total effective energy of the satellite, `s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s`."""
        return derivative(s.s, 1) ** 2 / 2 + s.Lz**2 / (2 * s.s**2) - 1 / s.s

    @staticmethod
    def _thrust_direction(s: SimpleNamespace) -> Expression:
        """Return the thrust `sigma`: inwards while the satellite falls, outwards while it rises, to the end energy."""
        return where(SatelliteElement._satellite_energy(s) < s.H_final, where(derivative(s.s, 1) < 0, 1, -1), 0)
