"""Rail segment on a sleeper, two masses joined by a spring and a damper and loaded by passing wheels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative, pulse_train
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RailSegmentElement(Element):
    """Rail segment on a sleeper, two masses joined by a spring and a damper and loaded by passing wheels.

    The element holds the wheel load and the ballast under the sleeper, and the model joins the rail and the sleeper by
    a `SpringDamperInteraction` of the stiffness `kr` and the damping `br`.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[object, StateSpec(order=2)] = 0.0,
        z_d1: float = 0.0,
        zs: Annotated[object, StateSpec(order=2)] = 0.0,
        zs_d1: float = 0.0,
        Mr: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        Ms: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 156.0,
        kr: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6000.0,
        br: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 300.0,
        ks: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1500.0,
        bs: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 520.0,
        k2: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[object, ParameterSpec()] = 1.75,
        beta: Annotated[object, ParameterSpec()] = -1.0,
        F: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        width: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        spacing: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.z_d1 = z_d1
        self.zs = zs
        self.zs_d1 = zs_d1
        self.Mr = Mr
        self.Ms = Ms
        self.kr = kr
        self.br = br
        self.ks = ks
        self.bs = bs
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.T = T
        self.width = width
        self.spacing = spacing

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the rail by the wheels and of the sleeper by the ballast, the pad apart."""
        return [
            RateLaw(s.z, RailSegmentElement._wheel_load(s) / s.Mr),
            RateLaw(
                s.zs,
                (-s.bs * derivative(s.zs, 1) - s.ks * s.zs - s.k2 * (s.alpha * s.zs**2 + s.beta * s.zs**3)) / s.Ms,
            ),
        ]

    @staticmethod
    def _wheel_load(s: SimpleNamespace) -> Expression:
        """Return the force of the two wheels of a bogie: two pulses per period, `spacing` apart."""
        return pulse_train(s.t, s.T, s.width, s.F) + pulse_train(s.t, s.T, s.width, s.F, start=s.spacing)
