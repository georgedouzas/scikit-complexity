"""Car-following models: cars on a road that react to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import cos, sqrt, tanh

from ....core.base import Domain, parameter
from ....core.model import Element, Interaction, Rate, System, delayed
from ....core.runtime import register_model

Car = Element.define(
    'Car',
    states={'x': 0.0, 'v': 0.0},
    laws=lambda s: [Rate(s.x, s.v)],
)

LeadCar = Element.define(
    'LeadCar',
    states={'x': 0.0, 'v': 1.0},
    parameters={
        'u': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'amplitude': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'omega': parameter(default=0.5, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [Rate(s.x, s.v), Rate(s.v, -s.u * s.amplitude * s.omega * cos(s.omega * s.t))],
)

OptimalVelocity = Interaction.define(
    'OptimalVelocity',
    roles=['leader', 'follower'],
    parameters={
        'a': parameter(default=1.0, domain=Domain.POSITIVE),
        'V0': parameter(default=1.0, domain=Domain.POSITIVE),
        'hc': parameter(default=2.0, domain=Domain.POSITIVE),
        'offset': 0.0,
    },
    laws=lambda s: [
        Rate(
            s.follower.v,
            s.a * (s.V0 * (tanh(s.leader.x - s.follower.x + s.offset - s.hc) + tanh(s.hc)) - s.follower.v),
        ),
    ],
)

IntelligentDriver = Interaction.define(
    'IntelligentDriver',
    roles=['leader', 'follower'],
    parameters={
        'v0': parameter(default=30.0, domain=Domain.POSITIVE),
        'T': parameter(default=1.5, domain=Domain.NONNEGATIVE),
        'a': parameter(default=1.0, domain=Domain.POSITIVE),
        'b': parameter(default=1.5, domain=Domain.POSITIVE),
        's0': parameter(default=2.0, domain=Domain.NONNEGATIVE),
        'delta': parameter(default=4.0, domain=Domain.POSITIVE),
        'length': parameter(default=5.0, domain=Domain.NONNEGATIVE),
        'offset': 0.0,
    },
    laws=lambda s: [
        Rate(
            s.follower.v,
            s.a
            * (
                1
                - (s.follower.v / s.v0) ** s.delta
                - (
                    (s.s0 + s.follower.v * s.T + s.follower.v * (s.follower.v - s.leader.v) / (2 * sqrt(s.a * s.b)))
                    / (s.leader.x - s.follower.x + s.offset - s.length)
                )
                ** 2
            ),
        ),
    ],
)

DelayedFollowing = Interaction.define(
    'DelayedFollowing',
    roles=['leader', 'follower'],
    parameters={
        'reaction': parameter(default=1.0, domain=Domain.POSITIVE),
        'T': parameter(default=1.0, domain=Domain.POSITIVE),
        'power': parameter(default=0.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(
            s.follower.v,
            s.reaction
            * (delayed(s.leader.v, s.T) - delayed(s.follower.v, s.T))
            / (delayed(s.leader.x, s.T) - delayed(s.follower.x, s.T)) ** s.power,
        ),
    ],
)


def _ring_positions(n_cars: int, length: float, perturbation: float) -> list[float]:
    """Return evenly spaced positions on a ring, with the first car moved forward by a perturbation."""
    positions = np.arange(n_cars) * length / n_cars
    positions[0] += perturbation
    return [float(position) for position in positions]


@register_model(
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
    """Return the optimal velocity model of cars on a ring road, where uniform flow breaks into stop-and-go waves.

    Car `i` relaxes its speed towards the optimal velocity of its headway `h`, `v_i' = a (V(h) - v_i)` with
    `V(h) = V0 (tanh(h - hc) + tanh(hc))`. Uniform flow with headway `b = length / n_cars` is unstable when
    `V'(b) > a / 2`, and then a jam forms that moves against the direction of travel.

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
            The model, with the positions `car_<i>.x` and speeds `car_<i>.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import optimal_velocity
        >>> len(optimal_velocity(n_cars=5).interactions)
        5
    """
    headway = length / n_cars
    speed = V0 * (np.tanh(headway - hc) + np.tanh(hc))
    positions = _ring_positions(n_cars, length, perturbation)
    elements = [Car(f'car_{index}', x=positions[index], v=float(speed)) for index in range(n_cars)]
    interactions = [
        OptimalVelocity(
            f'follow_{index}',
            leader=f'car_{(index + 1) % n_cars}',
            follower=f'car_{index}',
            a=a,
            V0=V0,
            hc=hc,
            offset=length if index == n_cars - 1 else 0.0,
        )
        for index in range(n_cars)
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'ode', 'car following'),
    references=('Treiber, Hennecke, and Helbing 2000', 'Chowdhury, Santen, and Schadschneider 2000'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def intelligent_driver_model(
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
    """Return the intelligent driver model of cars on a ring road.

    Car `i` accelerates as `v' = a (1 - (v / v0)^δ - (s* / s)^2)` with the gap `s` to the car ahead and the desired
    gap `s* = s0 + v T + v Δv / (2 sqrt(a b))`, where `Δv` is the speed difference to the car ahead. Cars start at rest.

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
            The model, with the positions `car_<i>.x` and speeds `car_<i>.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import intelligent_driver_model
        >>> intelligent_driver_model(n_cars=4).interactions[3].offset
        500.0
    """
    positions = _ring_positions(n_cars, length, perturbation)
    elements = [Car(f'car_{index}', x=positions[index], v=0.0) for index in range(n_cars)]
    interactions = [
        IntelligentDriver(
            f'follow_{index}',
            leader=f'car_{(index + 1) % n_cars}',
            follower=f'car_{index}',
            v0=v0,
            T=T,
            a=a,
            b=b,
            s0=s0,
            delta=delta,
            length=car_length,
            offset=length if index == n_cars - 1 else 0.0,
        )
        for index in range(n_cars)
    ]
    return System(elements=elements, interactions=interactions)


@register_model(
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
    along the chain when `ω < 2 λ sin(ω T)` for the linear model, `power = 0`.

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
            The power of the distance in the reaction, 0 for the linear model.

    Returns:
        system:
            The model, with the lead car `car_0` and the following cars `car_<j>`, each with position `x` and speed `v`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import delayed_car_following
        >>> delayed_car_following(n_cars=3).elements[2].x
        -20.0
    """
    lead = LeadCar('car_0', x=0.0, v=u, u=u, amplitude=amplitude, omega=omega)
    followers = [Car(f'car_{index}', x=-index * spacing, v=u) for index in range(1, n_cars)]
    interactions = [
        DelayedFollowing(
            f'follow_{index}',
            leader=f'car_{index - 1}',
            follower=f'car_{index}',
            reaction=reaction,
            T=T,
            power=power,
        )
        for index in range(1, n_cars)
    ]
    return System(elements=[lead, *followers], interactions=interactions)
