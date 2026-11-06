"""Relaxation of the speed of every car of a ring road towards the speed of its local density, with noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.interactions import RelaxationInteraction
from ....core.specs import ParameterSpec, RoleSpec
from ._optimal_velocity import OptimalVelocityInteraction


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
