"""Test the `_control` module of `skcomplex.systems.engineering.control`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.systems import System
from skcomplex.systems.engineering.control import (
    ConvectionInteraction,
    HeatCapacityInteraction,
    HeatedRoomSystem,
    PendulumSwingUpSystem,
    PidLoopSystem,
    SatelliteTransferSystem,
    ThermalConductorInteraction,
    ThermostatSystem,
)


def test_thermostat_settles_below_the_setting():
    """Test that the leaky room settles at `(setting + leak * outside) / (1 + leak)`."""
    trajectory = ThermostatSystem().prepare().simulate(until=48, dt=0.25)
    assert trajectory.get_values('room.T')[-1] == pytest.approx((18 + 0.1 * 10) / 1.1, rel=1e-4)


@pytest.mark.parametrize(('Ki', 'final'), [(1.0, 1.0), (0.0, 0.5)])
def test_pid_loop_integral_action(Ki, final):
    """Test that integral action removes the error left by a disturbance, which proportional control keeps."""
    trajectory = PidLoopSystem(Ki=Ki).prepare().simulate(until=40, dt=0.1)
    assert trajectory.get_values('process.y')[-1] == pytest.approx(final, abs=1e-3)


def test_pendulum_swing_up_reaches_the_top():
    """Test that the energy control brings the pendulum to its inverted position."""
    trajectory = PendulumSwingUpSystem(solver=RK4Solver(dt=0.01)).prepare().simulate(until=100, dt=0.1)
    theta, omega = trajectory.get_values('pendulum.theta')[-100:], trajectory.get_values("pendulum.theta'")[-100:]
    assert np.cos(theta) == pytest.approx(np.full(100, -1.0), abs=0.01)
    assert np.abs(omega[-10:]).max() < 0.01


def test_satellite_transfer_time():
    """Test that the transfer from the periapsis reaches the final energy after about 9.81."""
    trajectory = SatelliteTransferSystem(solver=RK4Solver(dt=0.005)).prepare().simulate(until=15, dt=0.05)
    energy = trajectory.get_values('energy')
    assert energy[0] == pytest.approx(-0.7, abs=1e-3)
    assert trajectory.times[np.argmax(energy >= -0.2)] == pytest.approx(9.81, abs=0.1)
    assert energy[-1] == pytest.approx(-0.2, abs=1e-3)


def test_heated_room_stays_within_the_band_of_its_thermostat():
    """Test that the room rises to the band, stays in it, and the heater switches on and off."""
    trajectory = HeatedRoomSystem().prepare().simulate(until=14400.0, dt=10.0)
    temperature = trajectory.get_values('room.T').ravel()
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
    assert set(np.unique(trajectory.get_values('heater.closed'))) == {0.0, 1.0}


def test_heated_room_records_the_power_of_its_heater():
    """Test that the catalog entry records the power of the heater, `V^2 / R` while its switch is closed."""
    trajectory = HeatedRoomSystem().prepare().simulate(until=200.0, dt=10.0)
    assert np.allclose(trajectory.get_values('power').ravel(), 230.0**2 / 25.0)


def test_body_cools_through_a_conductor_and_convection_in_closed_form():
    """Test that a body cools towards a fluid at the rate of the conductance in series with the convection."""
    system = System(
        elements=[Element('body', values={'T': 80.0}), Element('skin', values={'T': 0.0})],
        interactions=[
            HeatCapacityInteraction('capacity', node='body', C=4.0),
            ThermalConductorInteraction('wall', a='body', b='skin', G=2.0),
            ConvectionInteraction('surface', node='skin', h=2.0, fluid=20.0),
        ],
    )
    trajectory = system.prepare().simulate(until=5.0, dt=0.5)
    expected = 20.0 + 60.0 * np.exp(-trajectory.times * (2.0 * 2.0 / (2.0 + 2.0)) / 4.0)
    assert np.allclose(trajectory.get_values('body.T').ravel(), expected, rtol=1e-4)
    assert np.allclose(trajectory.get_values('skin.T').ravel(), (expected + 20.0) / 2, rtol=1e-4)
