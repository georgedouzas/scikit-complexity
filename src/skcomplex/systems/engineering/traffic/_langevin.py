"""Langevin traffic: cars that relax with noise to a speed set by the local density, and their density equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement
from ....core.components.interactions import RelaxationInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import divergence, laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver, RK4Solver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common._car import CarElement
from ..common._optimal_velocity import OptimalVelocityInteraction


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


class LangevinDrivingInteraction(RelaxationInteraction):
    """Relaxation of the speed of every car of a ring road towards the speed of its local density, with noise.

    The speed follows `dv = -gamma (v - u(rho)) dt + sqrt(2 D1 rho) dW`, with the desired speed
    `u(rho) = u0 (1 - rho / rho_max)` of the density `rho`, the inverse of the headway to the next car, an
    Ornstein-Uhlenbeck relaxation over the time `1 / gamma` with a noise that grows with the density.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of cars, in the order they drive.

        gamma:
            The relaxation rate.

        u0:
            The desired speed on an empty road.

        rho_max:
            The jam density.

        D1:
            The strength of the noise.

        road_length:
            The length of the ring road.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import LangevinDrivingInteraction
        >>> LangevinDrivingInteraction('drive', target='cars', gamma=2.0).state
        'v'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        u0: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        rho_max: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        road_length: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.gamma = gamma
        self.u0 = u0
        self.rho_max = rho_max
        self.D1 = D1
        self.road_length = road_length

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the desired speed of the local density of every car."""
        return LangevinDrivingInteraction._target_speed(
            LangevinDrivingInteraction._local_density(namespace.target.x, namespace.road_length),
            namespace,
        )

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time `1 / gamma`."""
        return 1 / namespace.gamma

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the amplitude `sqrt(2 D1 rho)` of the noise of the local density."""
        return sqrt(
            2 * namespace.D1 * LangevinDrivingInteraction._local_density(namespace.target.x, namespace.road_length),
        )

    @staticmethod
    def _target_speed(density: Expression, s: SimpleNamespace) -> Expression:
        """Return the desired speed `u(rho) = u0 (1 - rho / rho_max)` at a density."""
        return s.u0 * (1 - density / s.rho_max)

    @staticmethod
    def _local_density(x: Expression, road_length: Expression) -> Expression:
        """Return the density seen by every car of a ring road, the inverse of its headway."""
        return 1 / OptimalVelocityInteraction._headway(x, road_length)


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'sde', 'langevin', 'car following'),
    references=('Bertin 2021',),
    defaults={'until': 200.0, 'dt': 1.0, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
    `u0 rho^2 / rho_max < gamma / 2`, as for the optimal velocity system. Strong noise lets cars pass each other, and
    the headway of a car that has passed the car ahead wraps around the ring.

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
            The system, with the group `cars` of positions `cars.x` and speeds `cars.v`, car `i + 1` ahead of car `i`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import langevin_traffic
        >>> float(langevin_traffic(n_cars=10, length=20.0).elements[0].values['v'][0])
        0.5
    """
    density = n_cars / length
    positions = np.arange(n_cars) * length / n_cars
    cars = GroupElement(
        'cars',
        element=CarElement,
        size=n_cars,
        values={'x': positions, 'v': np.full(n_cars, u0 * (1 - density / rho_max))},
    )
    driving = LangevinDrivingInteraction(
        'drive',
        target='cars',
        gamma=gamma,
        u0=u0,
        rho_max=rho_max,
        D1=D1,
        road_length=length,
    )
    return System(elements=[cars], interactions=[driving])


@register_system(
    field='engineering/traffic',
    tags=('traffic', 'field', 'jams', 'instability', 'pde'),
    references=('Bertin 2021',),
    defaults={'until': 400.0, 'dt': 4.0, 'solver': RK4Solver(dt=0.1)},
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
            The system, with the density `road.rho` and the flux `road.w`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import traffic_density_equation
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
    road = FieldElement('road', element=TrafficMomentsElement, space=space, values=values)
    return System(elements=[road])
