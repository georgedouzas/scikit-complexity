"""Density `rho` and flow `w` of Langevin traffic on a road, the first two moments of its speeds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import divergence, laplacian
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._langevin_driving import LangevinDrivingInteraction


class TrafficMomentsElement(Element):
    """Density `rho` and flow `w` of Langevin traffic on a road, the first two moments of its speeds."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        w: Annotated[object, StateSpec()] = 0.25,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        u0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.w = w
        self.gamma = gamma
        self.u0 = u0
        self.rho_max = rho_max
        self.D1 = D1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the closed moment equations of the density and the flow."""
        return [
            RateLaw(s.rho, -divergence([s.w]) + laplacian(s.D1 * s.rho**2 / 2)),
            RateLaw(
                s.w,
                -divergence(
                    [
                        2 * LangevinDrivingInteraction._target_speed(s.rho, s) * s.w
                        - s.rho * LangevinDrivingInteraction._target_speed(s.rho, s) ** 2,
                    ],
                )
                - s.gamma * (s.w - s.rho * LangevinDrivingInteraction._target_speed(s.rho, s))
                + s.D1 * (laplacian(s.rho * s.w) + s.rho * laplacian(s.w) - s.w * laplacian(s.rho)) / 2,
            ),
        ]
