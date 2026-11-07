"""Test the `_control` module of `skcomplex.systems.engineering.control`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation, describe
from skcomplex.systems.engineering.control import (
    heated_room,
    minimum_time_control,
    pendulum_swing_up,
    pid_loop,
    satellite_transfer,
    thermostat,
)


def test_thermostat_settles_below_the_setting():
    """Test that the leaky room settles at `(setting + leak * outside) / (1 + leak)`."""
    trajectory = Simulation(thermostat(), dt=0.25).prepare().run(until=48).trajectory_
    assert trajectory.values('room.T')[-1] == pytest.approx((18 + 0.1 * 10) / 1.1, rel=1e-4)


@pytest.mark.parametrize(('Ki', 'final'), [(1.0, 1.0), (0.0, 0.5)])
def test_pid_loop_integral_action(Ki, final):
    """Test that integral action removes the error left by a disturbance, which proportional control keeps."""
    trajectory = Simulation(pid_loop(Ki=Ki), dt=0.1).prepare().run(until=40).trajectory_
    assert trajectory.values('process.y')[-1] == pytest.approx(final, abs=1e-3)


def test_pendulum_swing_up_reaches_the_top():
    """Test that the energy control brings the pendulum to its inverted position."""
    trajectory = Simulation(pendulum_swing_up(), dt=0.1, solver=RK4Solver(dt=0.01)).prepare().run(until=100).trajectory_
    theta, omega = trajectory.values('pendulum.theta')[-100:], trajectory.values('pendulum.theta_d1')[-100:]
    assert np.cos(theta) == pytest.approx(np.full(100, -1.0), abs=0.01)
    assert np.abs(omega[-10:]).max() < 0.01


def test_satellite_transfer_time():
    """Test that the transfer from the periapsis reaches the final energy after about 9.81."""
    trajectory = (
        Simulation(
            built := satellite_transfer(),
            dt=0.05,
            solver=RK4Solver(dt=0.005),
            observables=describe('satellite_transfer').observables_of(built),
        )
        .prepare()
        .run(until=15)
        .trajectory_
    )
    energy = trajectory.values('energy')
    assert energy[0] == pytest.approx(-0.7, abs=1e-3)
    assert trajectory.times[np.argmax(energy >= -0.2)] == pytest.approx(9.81, abs=0.1)
    assert energy[-1] == pytest.approx(-0.2, abs=1e-3)


def test_heated_room_stays_within_the_band_of_its_thermostat():
    """Test that the room rises to the band, stays in it, and the heater switches on and off."""
    trajectory = Simulation(heated_room(), dt=10.0).prepare().run(until=14400.0).trajectory_
    temperature = trajectory.values('room.T').ravel()
    reached = int(np.argmax(temperature >= 19.5))
    assert temperature[0] == pytest.approx(10.0)
    assert reached > 0
    assert np.all(np.diff(temperature[: reached + 1]) > 0)
    assert temperature[reached:].min() >= 19.5 - 1e-6
    assert temperature[reached:].max() <= 20.5 + 1e-6
    opened = [event for event in trajectory.events if event.label == 'open_switch']
    closed = [event for event in trajectory.events if event.label == 'close_switch']
    assert len(opened) >= 2
    assert len(closed) >= 2
    assert set(np.unique(trajectory.values('heater.on'))) == {0.0, 1.0}


def test_heated_room_records_the_power_of_its_heater():
    """Test that the catalog entry records the power of the heater, `V^2 / R` while its switch is closed."""
    trajectory = create_simulation('heated_room').set_params(dt=10.0).prepare().run(until=200.0).trajectory_
    assert np.allclose(trajectory.values('power').ravel(), 230.0**2 / 25.0)


def _distance_to_origin(trajectory):
    """Return the times and the distance of a controlled plant to the origin of its phase space."""
    x, y = np.ravel(trajectory.values('plant.x')), np.ravel(trajectory.values('plant.y'))
    return trajectory.times, np.hypot(x, y)


def test_minimum_time_control_brings_a_free_particle_to_the_origin_in_the_minimum_time():
    """Test that a free particle from `(3, 2)` reaches the origin near the time `v + 2 sqrt(x + v^2 / 2)` and stays."""
    trajectory = create_simulation('minimum_time_control', interval=0.01).prepare().run().trajectory_
    times, distance = _distance_to_origin(trajectory)
    arrival = times[np.argmax(distance < 0.1)]
    assert arrival == pytest.approx(2 + 2 * np.sqrt(5), abs=0.2)
    assert distance[times > arrival + 0.5].max() < 0.1


@pytest.mark.parametrize(('plant', 'x', 'v'), [('harmonic_oscillator', -2.5, 1.5), ('saddle', 0.5, -0.2)])
def test_minimum_time_control_brings_other_plants_to_the_origin(plant, x, v):
    """Test that the harmonic oscillator and a controllable state of the saddle reach the origin and stay near it."""
    trajectory = create_simulation('minimum_time_control', plant=plant, x=x, v=v, interval=0.01).prepare().run()
    times, distance = _distance_to_origin(trajectory.trajectory_)
    assert distance[times > 6].max() < 0.1


def test_minimum_time_control_cannot_hold_the_saddle_outside_its_controllable_strip():
    """Test that the saddle escapes from a state above the line `y = -(x - 1)`, whatever the control."""
    trajectory = create_simulation('minimum_time_control', plant='saddle', x=0.5, v=0.8).prepare().run(until=5)
    assert _distance_to_origin(trajectory.trajectory_)[1][-1] > 10


def test_minimum_time_control_rejects_an_unknown_plant():
    """Test that an unknown plant raises a parameter error."""
    with pytest.raises(ParameterError, match='Unknown plant'):
        minimum_time_control(plant='pendulum')
