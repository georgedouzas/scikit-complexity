"""Acceleration of every car of a ring road by the intelligent driver model of its speed and the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....core.laws.utils import shift_member
from ....specs import ParameterSpec, RoleSpec
from ._optimal_velocity import OptimalVelocityInteraction


class IntelligentDriverInteraction(RateInteraction):
    """Acceleration of every car of a ring road by the intelligent driver model of its speed and the car ahead.

    The speed follows `v' = a (1 - (v / v0)^δ - (s*(v, Δv) / (h - l))^2)`, with the desired gap
    `s* = s0 + v T + v Δv / (2 sqrt(a b))`, the approach rate `Δv` to the next car, and the headway `h` less the car
    length `l`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of cars, in the order they drive.

        v0:
            The desired speed.

        T:
            The safe time headway.

        a:
            The largest acceleration.

        b:
            The comfortable deceleration.

        s0:
            The jam distance.

        delta:
            The acceleration exponent.

        length:
            The length of a car.

        road_length:
            The length of the ring road.

    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        v0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        s0: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        delta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        length: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        road_length: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 500.0,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.delta = delta
        self.length = length
        self.road_length = road_length

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the acceleration of the intelligent driver model."""
        s, cars = namespace, namespace.target
        gap = s.s0 + cars.v * s.T + cars.v * (cars.v - shift_member(cars.v, 1)) / (2 * sqrt(s.a * s.b))
        return s.a * (
            1
            - (cars.v / s.v0) ** s.delta
            - (gap / (OptimalVelocityInteraction._headway(cars.x, s.road_length) - s.length)) ** 2
        )
