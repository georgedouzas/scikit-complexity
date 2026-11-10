"""Test the `_mechatronics` module of `skcomplex.systems.engineering.mechatronics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.mechatronics import DcMotorSystem, QuarterCarSystem


def test_dc_motor_settles_at_its_closed_form_speed():
    """Test that the angular velocity settles within 1% of `k V / (R b + k^2)`."""
    V, R, k, b = 12.0, 1.0, 0.5, 0.2
    trajectory = simulate(DcMotorSystem(V, R=R, k=k, b=b).prepare().system_, until=40.0, dt=0.05)
    speed = k * V / (R * b + k**2)
    assert trajectory.get_values('rotor.phi_d1')[-1] == pytest.approx(speed, rel=0.01)
    assert trajectory.get_values('winding.p_i')[-1] == pytest.approx(b * speed / k, rel=0.01)


def test_dc_motor_records_its_observables():
    """Test that the catalog entry of the motor records its speed and its current."""
    trajectory = simulate(DcMotorSystem(), until=20.0, dt=0.1)
    assert trajectory.get_values('speed')[-1] == pytest.approx(trajectory.get_values('rotor.phi_d1')[-1])
    assert trajectory.get_values('current')[-1] == pytest.approx(trajectory.get_values('winding.p_i')[-1])
    assert DcMotorSystem.field == 'engineering/mechatronics'


def test_quarter_car_settles_at_the_step_height():
    """Test that the body and the wheel settle at the step height and the energy of the springs decays."""
    trajectory = simulate(QuarterCarSystem(height=0.05).prepare().system_, until=10.0, dt=0.01)
    assert trajectory.get_values('car.y1')[-1] == pytest.approx(0.05, abs=1e-4)
    assert trajectory.get_values('car.y2')[-1] == pytest.approx(0.05, abs=1e-4)
    settled = trajectory.times > 0.5
    suspension = 16000.0 * (trajectory.get_values('car.y1') - trajectory.get_values('car.y2')).ravel() ** 2 / 2
    tire = 160000.0 * (trajectory.get_values('car.y2').ravel() - 0.05) ** 2 / 2
    energy = (suspension + tire)[settled]
    assert energy[-1] < 1e-3 * energy[:100].max()
    assert np.mean(energy[-100:]) < np.mean(energy[:100])


def test_quarter_car_records_its_energy():
    """Test that the catalog entry of the quarter car records the energy of its springs."""
    energy = simulate(QuarterCarSystem(), until=5.0, dt=0.05).get_values('energy').ravel()
    assert energy[0] == 0.0
    assert energy[-1] < 1e-3 * energy.max()
