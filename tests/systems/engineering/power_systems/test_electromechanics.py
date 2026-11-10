"""Test the `_electromechanics` module of `skcomplex.systems.engineering.power_systems`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.power_systems import (
    FaradayDiskSystem,
    RegenerativeBrakingSystem,
    WheelRailContactSystem,
)


def test_regenerative_braking_stores_energy():
    """Test that braking charges the capacitor, slows the car, and never creates energy."""
    trajectory = simulate(RegenerativeBrakingSystem().prepare().system_, until=30, dt=0.5)
    speed, current, voltage = (
        trajectory.get_values('car.v'),
        trajectory.get_values('car.I'),
        trajectory.get_values('car.Vs'),
    )
    energy = 1000.0 * speed**2 / 2 + 680.0 * voltage**2 / 2 + 0.01 * current**2 / 2
    assert np.all(np.diff(energy) < 1e-6)
    assert speed[-1] < speed[0] / 2
    assert voltage[-1] > 5.0


def test_faraday_disk_reaches_torque_balance():
    """Test that the charger settles where the drive torque balances friction and the braking Lorentz torque."""
    trajectory = simulate(FaradayDiskSystem().prepare().system_, until=50, dt=0.1)
    current, voltage, omega = (trajectory.get_values(f'charger.{name}')[-1] for name in ('I', 'U', 'omega'))
    coupling = (1.5 + 1.2566370614359173e-4 * 250 * current / 0.05) * (0.55**2 - 0.025**2) / 2
    assert 0.5 + 0.5 * 0.55 * omega + current * coupling == pytest.approx(4.8, abs=1e-4)
    assert voltage == pytest.approx(109.0 * current, abs=1e-4)


def test_faraday_disk_charges_only_above_the_battery_voltage():
    """Test that the default disk discharges the battery steadily and a fast disk makes the current oscillate."""
    slow = simulate(FaradayDiskSystem().prepare().system_, until=50, dt=0.1)
    fast = simulate(FaradayDiskSystem(M=11.0).prepare().system_, until=200, dt=0.1)
    current, omega = slow.get_values('charger.I')[-1], slow.get_values('charger.omega')[-1]
    assert current == pytest.approx(-0.040, abs=1e-3)
    assert 1.5 * (0.55**2 - 0.025**2) / 2 * omega < 8.0
    assert np.ptp(fast.get_values('charger.I')[-500:]) > 1.0


def test_wheel_rail_response_is_periodic():
    """Test that the rail settles on a response with the period of the passing bogies."""
    trajectory = simulate(WheelRailContactSystem().prepare().system_, until=60, dt=0.05)
    rail = trajectory.get_values('rail.z')
    assert rail[-1] == pytest.approx(rail[-201], abs=1e-4)
    assert rail.max() > 0.1
