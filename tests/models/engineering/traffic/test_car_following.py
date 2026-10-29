"""Test the `_car_following` module of `skcomplex.models.engineering.traffic`."""

import math
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sqrt, tanh

from skcomplex.core.elements import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.engineering.traffic import (
    delayed_car_following,
    intelligent_driver_model,
    optimal_velocity,
)
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


def _cars(trajectory, n_cars, variable):
    """Return a variable of every car as an array of shape (frames, cars)."""
    return np.array([trajectory.values(f'car_{index}.{variable}') for index in range(n_cars)]).T


class CarElement(Element):
    """Car element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        v: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        """Return the laws of the car element."""
        return [RateLaw(s.x, s.v)]


class PerCarVelocityInteraction(Interaction):
    """Per car velocity interaction."""

    def __init__(
        self,
        label: str,
        *,
        leader: Annotated[object, RoleSpec()],
        follower: Annotated[object, RoleSpec()],
        a: Annotated[object, ParameterSpec()] = 1.0,
        V0: Annotated[object, ParameterSpec()] = 1.0,
        hc: Annotated[object, ParameterSpec()] = 2.0,
        offset: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.leader = leader
        self.follower = follower
        self.a = a
        self.V0 = V0
        self.hc = hc
        self.offset = offset

    def _get_laws(self, s):
        """Return the laws of the per car velocity interaction."""
        return [
            RateLaw(
                s.follower.v,
                s.a * (s.V0 * (tanh(s.leader.x - s.follower.x + s.offset - s.hc) + tanh(s.hc)) - s.follower.v),
            ),
        ]


class PerCarDriverInteraction(Interaction):
    """Per car driver interaction."""

    def __init__(
        self,
        label: str,
        *,
        leader: Annotated[object, RoleSpec()],
        follower: Annotated[object, RoleSpec()],
        v0: Annotated[object, ParameterSpec()] = 30.0,
        T: Annotated[object, ParameterSpec()] = 1.5,
        a: Annotated[object, ParameterSpec()] = 1.0,
        b: Annotated[object, ParameterSpec()] = 1.5,
        s0: Annotated[object, ParameterSpec()] = 2.0,
        delta: Annotated[object, ParameterSpec()] = 4.0,
        length: Annotated[object, ParameterSpec()] = 5.0,
        offset: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.leader = leader
        self.follower = follower
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.delta = delta
        self.length = length
        self.offset = offset

    def _get_laws(self, s):
        """Return the laws of the per car driver interaction."""
        return [
            RateLaw(
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
        ]


def _per_car(system, interaction):
    """Return the ring road of a group of cars rebuilt as one element and one interaction per car, as before."""
    cars, following = system.elements[0], system.interactions[0]
    n_cars, road = cars.size, following.road_length
    elements = [
        CarElement(
            f'car_{index}',
            x=cars.values['x'][index],
            v=float(np.broadcast_to(cars.values['v'], n_cars)[index]),
        )
        for index in range(n_cars)
    ]
    settings = {
        key: value for key, value in following.get_params().items() if key not in {'label', 'cars', 'road_length'}
    }
    interactions = [
        interaction(
            f'follow_{index}',
            leader=f'car_{(index + 1) % n_cars}',
            follower=f'car_{index}',
            offset=road if index == n_cars - 1 else 0.0,
            **settings,
        )
        for index in range(n_cars)
    ]
    return System(elements=elements, interactions=interactions)


def test_optimal_velocity_jams_move_backwards():
    """Test that uniform flow above the critical density breaks into a stop-and-go wave that moves backwards."""
    n_cars, length = 12, 24.0
    trajectory = Simulation(optimal_velocity(n_cars, length), dt=1.0).check().prepare().run(until=150).trajectory_
    positions, speeds = trajectory.values('cars.x'), trajectory.values('cars.v')
    assert np.ptp(speeds[0]) < 0.01
    assert speeds[-1].min() < 0.2
    assert speeds[-1].max() > 1.5
    jam = np.array([positions[frame, np.argmin(speeds[frame])] % length for frame in range(100, 151)])
    shifts = (np.diff(jam) + length / 2) % length - length / 2
    assert shifts.sum() < -5.0
    assert speeds[100:].mean() > 0.5


def test_optimal_velocity_below_critical_density_stays_uniform():
    """Test that a perturbed uniform flow with large headways relaxes back to uniform flow."""
    n_cars, length = 8, 32.0
    trajectory = Simulation(optimal_velocity(n_cars, length), dt=1.0).check().prepare().run(until=80).trajectory_
    speeds = trajectory.values('cars.v')
    assert np.ptp(speeds[-1]) < 0.01


def test_intelligent_drivers_reach_equilibrium_speed():
    """Test that cars at low density reach the equilibrium speed of their gap, `s = s*(v) / sqrt(1 - (v / v0)^4)`."""
    trajectory = Simulation(intelligent_driver_model(10, 250.0), dt=1.0).check().prepare().run(until=100).trajectory_
    speed = trajectory.values('cars.v')[-1].mean()
    gap = 250.0 / 10 - 5.0
    assert gap == pytest.approx((2.0 + speed * 1.5) / math.sqrt(1 - (speed / 30.0) ** 4), rel=1e-3)


def test_ring_roads_hold_one_group_of_cars():
    """Test that a ring road holds one group and one interaction whatever the number of cars."""
    system = optimal_velocity(n_cars=50)
    assert [element.size for element in system.elements] == [50]
    assert len(system.interactions) == 1
    trajectory = Simulation(system, dt=1.0).check().prepare().run(until=2).trajectory_
    assert trajectory.values('cars.x').shape == trajectory.values('cars.v').shape == (3, 50)


@pytest.mark.parametrize(
    ('factory', 'interaction', 'until'),
    [(optimal_velocity, PerCarVelocityInteraction, 200.0), (intelligent_driver_model, PerCarDriverInteraction, 100.0)],
)
def test_ring_roads_match_one_interaction_per_car(factory, interaction, until):
    """Test that the ring road of one group runs as the ring of one element and one interaction per car under RK45."""
    system = factory(n_cars=20)
    group = (
        Simulation(system, dt=1.0, solver=ScipyODESolver(method='RK45')).check().prepare().run(until=until).trajectory_
    )
    per_car = (
        Simulation(_per_car(system, interaction), dt=1.0, solver=ScipyODESolver(method='RK45'))
        .check()
        .prepare()
        .run(until=until)
    )
    per_car = per_car.trajectory_
    for variable in ('x', 'v'):
        np.testing.assert_allclose(
            group.values(f'cars.{variable}'),
            _cars(per_car, 20, variable),
            rtol=1e-8,
            atol=1e-12,
        )


@pytest.mark.parametrize(('reaction', 'grows'), [(1.0, True), (0.3, False)])
def test_delayed_car_following_instability(reaction, grows):
    """Test that oscillations grow along the chain when `omega < 2 reaction sin(omega T)` and decay otherwise."""
    trajectory = (
        Simulation(delayed_car_following(5, reaction=reaction), dt=0.5).check().prepare().run(until=60).trajectory_
    )
    amplitudes = np.ptp(_cars(trajectory, 5, 'v')[60:], axis=0)
    assert (amplitudes[-1] > amplitudes[0]) == grows
