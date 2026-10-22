"""Langevin traffic: cars that relax with noise to a speed set by the local density, and their density equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, Expression, parameter, state
from ....core.model import (
    Element,
    Field,
    Interaction,
    LatticeSpace,
    Noise,
    Rate,
    Symbols,
    System,
    divergence,
    laplacian,
)
from ....core.runtime import register_model
from ....core.solvers import RK4, EulerMaruyama
from ._car_following import Car, _headway


def _target_speed(density: Expression, s: Symbols) -> Expression:
    """Return the desired speed `u(rho) = u0 (1 - rho / rho_max)` at a density."""
    return s.u0 * (1 - density / s.rho_max)


def _local_density(s: Symbols) -> Expression:
    """Return the density seen by every car of a ring road, the inverse of its headway."""
    return 1 / _headway(s.cars.x, s.road_length)


LangevinDriving = Interaction.define(
    'LangevinDriving',
    roles=['cars'],
    parameters={
        'gamma': parameter(default=3.0, domain=Domain.POSITIVE),
        'u0': parameter(default=1.0, domain=Domain.POSITIVE),
        'rho_max': parameter(default=1.0, domain=Domain.POSITIVE),
        'D1': parameter(default=0.05, domain=Domain.NONNEGATIVE),
        'road_length': parameter(default=100.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Rate(s.cars.v, -s.gamma * (s.cars.v - _target_speed(_local_density(s), s))),
        Noise(s.cars.v, sqrt(2 * s.D1 * _local_density(s))),
    ],
)

TrafficMoments = Element.define(
    'TrafficMoments',
    states={'rho': state(default=0.5, domain=Domain.NONNEGATIVE), 'w': 0.25},
    parameters={
        'gamma': parameter(default=1.0, domain=Domain.POSITIVE),
        'u0': parameter(default=1.0, domain=Domain.POSITIVE),
        'rho_max': parameter(default=1.0, domain=Domain.POSITIVE),
        'D1': parameter(default=0.2, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.rho, -divergence([s.w]) + laplacian(s.D1 * s.rho**2 / 2)),
        Rate(
            s.w,
            -divergence([2 * _target_speed(s.rho, s) * s.w - s.rho * _target_speed(s.rho, s) ** 2])
            - s.gamma * (s.w - s.rho * _target_speed(s.rho, s))
            + s.D1 * (laplacian(s.rho * s.w) + s.rho * laplacian(s.w) - s.w * laplacian(s.rho)) / 2,
        ),
    ],
)


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'sde', 'langevin', 'car following'),
    references=('Bertin 2021',),
    defaults={'until': 200.0, 'dt': 1.0, 'solver': EulerMaruyama(dt=0.01)},
)
def langevin_traffic(
    n_cars: int = 50,
    length: float = 100.0,
    *,
    gamma: float = 3.0,
    u0: float = 1.0,
    rho_max: float = 1.0,
    D1: float = 0.05,
) -> System:
    """Return cars on a ring road whose speeds relax with noise towards a desired speed set by the local density.

    Car `i` follows `dv_i = -gamma (v_i - u(rho_i)) dt + sqrt(2 D(rho_i)) dW` with the desired speed
    `u(rho) = u0 (1 - rho / rho_max)` and the noise intensity `D(rho) = D1 rho`, where the local density `rho_i` is the
    inverse of the headway to the car ahead. Cars start evenly spaced at the speed `u(rho)` of the mean density
    `rho = n_cars / length`. The mean speed stays at `u(rho)`, so the mean flow follows the fundamental diagram
    `rho u(rho)`, and the speeds fluctuate with the variance `D1 rho / gamma`. The uniform flow is linearly stable while
    `u0 rho^2 / rho_max < gamma / 2`, as for the optimal velocity model. Strong noise lets cars pass each other, and the
    headway of a car that has passed the car ahead wraps around the ring.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        gamma:
            The relaxation rate of the speed.

        u0:
            The desired speed on an empty road.

        rho_max:
            The jam density, at which the desired speed vanishes.

        D1:
            The noise intensity per unit density.

    Returns:
        system:
            The model, with the group `cars` of positions `cars.x` and speeds `cars.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import langevin_traffic
        >>> float(langevin_traffic(n_cars=10, length=20.0).elements[0].values['v'][0])
        0.5
    """
    density = n_cars / length
    positions = np.arange(n_cars) * length / n_cars
    cars = Car.group('cars', n_cars, x=positions, v=np.full(n_cars, u0 * (1 - density / rho_max)))
    driving = LangevinDriving(
        'drive',
        cars='cars',
        gamma=gamma,
        u0=u0,
        rho_max=rho_max,
        D1=D1,
        road_length=length,
    )
    return System(components=[cars], interactions=[driving])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'field', 'jams', 'instability', 'pde'),
    references=('Bertin 2021',),
    defaults={'until': 400.0, 'dt': 4.0, 'solver': RK4(dt=0.1)},
)
def traffic_density_equation(
    n: int = 100,
    *,
    length: float = 50.0,
    density: float = 0.5,
    gamma: float = 1.0,
    u0: float = 1.0,
    rho_max: float = 1.0,
    D1: float = 0.2,
    perturbation: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the density and flux equations of Langevin traffic on a ring road, whose uniform flow breaks into jams.

    The density `rho` and the flux `w` of cars that relax to the desired speed `u(rho) = u0 (1 - rho / rho_max)` at
    the rate `gamma`, with the diffusion coefficient `D(rho) = D1 rho`, follow the moment equations
    `rho_t + w_x = (D rho_x)_x` and `w_t + (2 u w - rho u^2)_x = -gamma (w - rho u) + (D w_x)_x`, closed by expanding
    the velocity distribution around `u(rho)`. A long density wave of the uniform flow grows when the effective
    diffusion coefficient `D(rho) + ((rho u^2)' - (rho u)'^2) / gamma = D(rho) - rho^2 u'(rho)^2 / gamma` is negative,
    so the flow is unstable above the density `D1 gamma rho_max^2 / u0^2`, and the perturbation grows into dense jams.
    The uniform density starts with independent Gaussian perturbations of relative size
    `perturbation` in every cell, at the flux `rho u(rho)`.

    Args:
        n:
            The number of cells.

        length:
            The length of the ring road.

        density:
            The mean density.

        gamma:
            The relaxation rate of the speed.

        u0:
            The desired speed on an empty road.

        rho_max:
            The jam density, at which the desired speed vanishes.

        D1:
            The diffusion coefficient per unit density.

        perturbation:
            The relative standard deviation of the initial density perturbations.

        random_state:
            The seed or generator of the initial perturbations.

    Returns:
        system:
            The model, with the density `road.rho` and the flux `road.w`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import traffic_density_equation
        >>> traffic_density_equation(10).elements[0].values['D1']
        0.2
    """
    rng = np.random.default_rng(random_state)
    rho = density * (1 + perturbation * rng.standard_normal(n))
    values = {
        'rho': rho,
        'w': rho * u0 * (1 - rho / rho_max),
        'gamma': gamma,
        'u0': u0,
        'rho_max': rho_max,
        'D1': D1,
    }
    space = LatticeSpace(n, spacing=length / n)
    road = Field('road', TrafficMoments, space, values=values)
    return System(components=[road])
