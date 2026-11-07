"""Car-following systems: cars on a road that react to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed, shift_member
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common._car import CarElement
from ..common._optimal_velocity import OptimalVelocityInteraction


class LeadCarElement(Element):
    """Lead car whose speed swings about a cruising speed `u` with an amplitude and a frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(role='position')] = 0.0,
        v: Annotated[object, StateSpec(role='velocity')] = 1.0,
        u: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        amplitude: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.u = u
        self.amplitude = amplitude
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the advance of the position and the forced swing of the speed."""
        return [RateLaw(s.x, s.v), RateLaw(s.v, -s.u * s.amplitude * s.omega * cos(s.omega * s.t))]


class DelayedFollowingInteraction(RateInteraction):
    """Acceleration of a follower car by the speed difference to its leader, both seen a reaction time late.

    The speed of the follower follows the Gazis-Herman-Rothery law
    `v_f'(t) = λ (v_l - v_f)(t - T) / (x_l - x_f)(t - T)^m` of the sensitivity `λ`, the reaction time `T`, and the
    power `m` of the gap.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the follower car.

        leader:
            The path of the car ahead of the follower.

        reaction:
            The sensitivity `λ`.

        T:
            The reaction time.

        power:
            The power `m` of the gap.

    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        leader: Annotated[str, RoleSpec()],
        reaction: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        power: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.leader = leader
        self.reaction = reaction
        self.T = T
        self.power = power

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the delayed response of the follower to its leader."""
        s, leader, follower = namespace, namespace.leader, namespace.target
        gap = delayed(leader.x, s.T) - delayed(follower.x, s.T)
        return s.reaction * (delayed(leader.v, s.T) - delayed(follower.v, s.T)) / gap**s.power


class IntelligentDriverInteraction(RateInteraction):
    """Acceleration of every car of a ring road by the intelligent driver system of its speed and the car ahead.

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
        """Return the acceleration of the intelligent driver system."""
        s, cars = namespace, namespace.target
        gap = s.s0 + cars.v * s.T + cars.v * (cars.v - shift_member(cars.v, 1)) / (2 * sqrt(s.a * s.b))
        return s.a * (
            1
            - (cars.v / s.v0) ** s.delta
            - (gap / (OptimalVelocityInteraction._headway(cars.x, s.road_length) - s.length)) ** 2
        )


def _ring_positions(n_cars: int, length: float, perturbation: float) -> list[float]:
    """Return evenly spaced positions on a ring, with the first car moved forward by a perturbation."""
    positions = np.arange(n_cars) * length / n_cars
    positions[0] += perturbation
    return [float(position) for position in positions]


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'ode', 'car following'),
    references=('Bando et al. 1995', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def optimal_velocity(
    n_cars: int = 20,
    length: float = 40.0,
    *,
    a: float = 1.0,
    V0: float = 1.0,
    hc: float = 2.0,
    perturbation: float = 0.1,
) -> System:
    """Return the optimal velocity system of cars on a ring road, where uniform flow breaks into stop-and-go waves.

    Car `i` relaxes its speed towards the optimal velocity of its headway `h`, `v_i' = a (V(h) - v_i)` with
    `V(h) = V0 (tanh(h - hc) + tanh(hc))`. Uniform flow with headway `b = length / n_cars` is unstable when
    `V'(b) > a / 2`, and then a jam forms that moves against the direction of travel. The cars are one group, and the
    headway of car `i` is the distance along the ring to member `i + 1`.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        a:
            The sensitivity of drivers.

        V0:
            The scale of the optimal velocity, which tends to `V0 (1 + tanh(hc))` on an empty road.

        hc:
            The headway at which the optimal velocity changes fastest.

        perturbation:
            The forward shift of the first car from uniform spacing.

    Returns:
        system:
            The system, with the group `cars` of positions `cars.x` and speeds `cars.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import optimal_velocity
        >>> system = optimal_velocity(n_cars=5)
        >>> system.elements[0].size, len(system.interactions)
        (5, 1)
    """
    headway = length / n_cars
    speed = V0 * (np.tanh(headway - hc) + np.tanh(hc))
    cars = GroupElement(
        'cars',
        element=CarElement,
        size=n_cars,
        values={'x': _ring_positions(n_cars, length, perturbation), 'v': float(speed)},
    )
    following = OptimalVelocityInteraction('follow', target='cars', a=a, V0=V0, hc=hc, road_length=length)
    return System(elements=[cars], interactions=[following])


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'ode', 'car following'),
    references=('Treiber, Hennecke, and Helbing 2000', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def intelligent_driver_system(
    n_cars: int = 20,
    length: float = 500.0,
    *,
    v0: float = 30.0,
    T: float = 1.5,
    a: float = 1.0,
    b: float = 1.5,
    s0: float = 2.0,
    delta: float = 4.0,
    car_length: float = 5.0,
    perturbation: float = 1.0,
) -> System:
    """Return the intelligent driver system of cars on a ring road.

    Car `i` accelerates as `v' = a (1 - (v / v0)^δ - (s* / s)^2)` with the gap `s` to the car ahead and the desired
    gap `s* = s0 + v T + v Δv / (2 sqrt(a b))`, where `Δv` is the speed difference to the car ahead. Cars start at rest.
    The cars are one group, and the car ahead of car `i` is member `i + 1` along the ring.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

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

        car_length:
            The length of a car, subtracted from the headway to give the gap.

        perturbation:
            The forward shift of the first car from uniform spacing.

    Returns:
        system:
            The system, with the group `cars` of positions `cars.x` and speeds `cars.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import intelligent_driver_system
        >>> intelligent_driver_system(n_cars=4).interactions[0].road_length
        500.0
    """
    cars = GroupElement(
        'cars',
        element=CarElement,
        size=n_cars,
        values={'x': _ring_positions(n_cars, length, perturbation), 'v': 0.0},
    )
    following = IntelligentDriverInteraction(
        'follow',
        target='cars',
        v0=v0,
        T=T,
        a=a,
        b=b,
        s0=s0,
        delta=delta,
        length=car_length,
        road_length=length,
    )
    return System(elements=[cars], interactions=[following])


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'delay', 'car following', 'instability'),
    references=('Gros 2015', 'Chandler, Herman, and Montroll 1958'),
    defaults={'until': 60.0, 'dt': 0.5},
)
def delayed_car_following(
    n_cars: int = 8,
    *,
    spacing: float = 10.0,
    u: float = 1.0,
    amplitude: float = 0.1,
    omega: float = 0.5,
    reaction: float = 1.0,
    T: float = 1.0,
    power: float = 0.0,
) -> System:
    """Return a chain of cars that react to the speed difference to the car ahead after a reaction time.

    Car `j + 1` follows car `j` as `x''_{j+1}(t) = λ Δv(t - T) / Δx(t - T)^power`, where `Δv` and `Δx` are the speed
    and position differences to the car ahead. The lead car drives at `u (1 - amplitude sin(ω t))`. Oscillations grow
    along the chain when `ω < 2 λ sin(ω T)` for the linear system, `power = 0`.

    Args:
        n_cars:
            The number of cars, the lead car included.

        spacing:
            The initial distance between neighbouring cars.

        u:
            The mean speed of the lead car.

        amplitude:
            The relative amplitude of the speed oscillation of the lead car.

        omega:
            The angular frequency of the speed oscillation of the lead car.

        reaction:
            The reaction constant `λ`.

        T:
            The reaction time.

        power:
            The power of the distance in the reaction, 0 for the linear system.

    Returns:
        system:
            The system, with the lead car `car_0` and the cars `car_<j>` behind it, with positions `x` and speeds `v`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import delayed_car_following
        >>> delayed_car_following(n_cars=3).elements[2].x
        -20.0
    """
    lead = LeadCarElement('car_0', x=0.0, v=u, u=u, amplitude=amplitude, omega=omega)
    followers = [CarElement(f'car_{index}', x=-index * spacing, v=u) for index in range(1, n_cars)]
    interactions = [
        DelayedFollowingInteraction(
            f'follow_{index}',
            target=f'car_{index}',
            leader=f'car_{index - 1}',
            reaction=reaction,
            T=T,
            power=power,
        )
        for index in range(1, n_cars)
    ]
    return System(elements=[lead, *followers], interactions=interactions)
