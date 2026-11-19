"""Test the `_car_following` module of `skcomplex.systems.engineering.traffic`."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sqrt, tanh

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.engineering.common.traffic import CarElement
from skcomplex.systems.engineering.traffic import (
    DelayedCarFollowingSystem,
    IntelligentDriverSystem,
    OptimalVelocitySystem,
)


def _cars(trajectory, n_cars, variable):
    """Return a variable of every car as an array of shape (frames, cars)."""
    return np.array([trajectory.get_values(f'car_{index}.{variable}') for index in range(n_cars)]).T


class PerCarVelocityInteraction(Interaction):
    """Per car velocity interaction."""

    def __init__(self, label: str, *, leader: Role, follower: Role, a=1.0, V0=1.0, hc=2.0, offset=0.0) -> None:
        super().__init__(label)
        self.leader = leader
        self.follower = follower
        self.a = a
        self.V0 = V0
        self.hc = hc
        self.offset = offset

    def _prepare(self, s):
        """Set the laws of the per car velocity interaction."""
        super()._prepare(s)
        if isinstance(s, SimpleNamespace):
            t, leader, follower = s.t, s.leader.x, s.follower.x
            optimal = s.V0 * (tanh(leader - follower + s.offset - s.hc) + tanh(s.hc))
            self.laws_ = [RateLaw(follower.diff(t, 2), s.a * (optimal - follower.diff(t)))]


class PerCarDriverInteraction(Interaction):
    """Per car driver interaction."""

    def __init__(
        self,
        label: str,
        *,
        leader: Role,
        follower: Role,
        v0=30.0,
        T=1.5,
        a=1.0,
        b=1.5,
        s0=2.0,
        delta=4.0,
        length=5.0,
        offset=0.0,
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

    def _prepare(self, s):
        """Set the laws of the per car driver interaction."""
        super()._prepare(s)
        if isinstance(s, SimpleNamespace):
            t, leader, follower = s.t, s.leader.x, s.follower.x
            v, ahead = follower.diff(t), leader.diff(t)
            gap = s.s0 + v * s.T + v * (v - ahead) / (2 * sqrt(s.a * s.b))
            headway = leader - follower + s.offset - s.length
            self.laws_ = [RateLaw(follower.diff(t, 2), s.a * (1 - (v / s.v0) ** s.delta - (gap / headway) ** 2))]


def _per_car(system, interaction):
    """Return the ring road of a group of cars rebuilt as one element and one interaction per car."""
    cars, following = system.elements[0], system.interactions[0]
    n_cars, road = cars.size, following.road_length
    speeds = np.broadcast_to((cars.values or {}).get("x'", 0.0), n_cars)
    elements = [
        CarElement(f'car_{index}', cars.x[index], values={"x'": float(speeds[index])}) for index in range(n_cars)
    ]
    settings = {
        key: value for key, value in following.get_params().items() if key not in {'label', 'target', 'road_length'}
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
    return System(elements=elements, interactions=interactions, solver=ScipyODESolver(method='RK45'))


def test_optimal_velocity_jams_move_backwards():
    """Test that uniform flow above the critical density breaks into a stop-and-go wave that moves backwards."""
    n_cars, length = 12, 24.0
    trajectory = OptimalVelocitySystem(n_cars, length).prepare().simulate(until=150, dt=1.0)
    positions, speeds = trajectory.get_values('cars.x'), trajectory.get_values("cars.x'")
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
    trajectory = OptimalVelocitySystem(n_cars, length).prepare().simulate(until=80, dt=1.0)
    speeds = trajectory.get_values("cars.x'")
    assert np.ptp(speeds[-1]) < 0.01


def test_intelligent_drivers_reach_equilibrium_speed():
    """Test that cars at low density reach the equilibrium speed of their gap, `s = s*(v) / sqrt(1 - (v / v0)^4)`."""
    trajectory = IntelligentDriverSystem(10, 250.0).prepare().simulate(until=100, dt=1.0)
    speed = trajectory.get_values("cars.x'")[-1].mean()
    gap = 250.0 / 10 - 5.0
    assert gap == pytest.approx((2.0 + speed * 1.5) / math.sqrt(1 - (speed / 30.0) ** 4), rel=1e-3)


def test_ring_roads_hold_one_group_of_cars():
    """Test that a ring road holds one group and one interaction whatever the number of cars."""
    system = OptimalVelocitySystem(n_cars=50).prepare().system_
    assert [element.size for element in system.elements] == [50]
    assert len(system.interactions) == 1
    trajectory = system.simulate(until=2, dt=1.0)
    assert trajectory.get_values('cars.x').shape == trajectory.get_values("cars.x'").shape == (3, 50)


@pytest.mark.parametrize(
    ('factory', 'interaction', 'until'),
    [
        (OptimalVelocitySystem, PerCarVelocityInteraction, 200.0),
        (IntelligentDriverSystem, PerCarDriverInteraction, 100.0),
    ],
)
def test_ring_roads_match_one_interaction_per_car(factory, interaction, until):
    """Test that the ring road of one group runs as the ring of one element and one interaction per car under RK45."""
    system = factory(n_cars=20).prepare().system_
    group = system.set_params(solver=ScipyODESolver(method='RK45')).prepare().simulate(until=until, dt=1.0)
    per_car = _per_car(system, interaction).prepare().simulate(until=until, dt=1.0)
    for variable in ('x', "x'"):
        np.testing.assert_allclose(
            group.get_values(f'cars.{variable}'),
            _cars(per_car, 20, variable),
            rtol=1e-8,
            atol=1e-12,
        )


@pytest.mark.parametrize(('reaction', 'grows'), [(1.0, True), (0.3, False)])
def test_delayed_car_following_instability(reaction, grows):
    """Test that oscillations grow along the chain when `omega < 2 reaction sin(omega T)` and decay otherwise."""
    trajectory = DelayedCarFollowingSystem(5, reaction=reaction).prepare().simulate(until=60, dt=0.5)
    amplitudes = np.ptp(_cars(trajectory, 5, "x'")[60:], axis=0)
    assert (amplitudes[-1] > amplitudes[0]) == grows
