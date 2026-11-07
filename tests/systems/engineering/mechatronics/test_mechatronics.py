"""Test the `_mechatronics` module of `skcomplex.systems.engineering.mechatronics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_simulation, describe
from skcomplex.systems.engineering.mechatronics import dc_motor, quarter_car


def test_dc_motor_settles_at_its_closed_form_speed():
    """Test that the angular velocity settles within 1% of `k V / (R b + k^2)`."""
    V, R, k, b = 12.0, 1.0, 0.5, 0.2
    trajectory = Simulation(dc_motor(V, R=R, k=k, b=b), dt=0.05).prepare().run(until=40.0).trajectory_
    speed = k * V / (R * b + k**2)
    assert trajectory.values('rotor.phi_d1')[-1] == pytest.approx(speed, rel=0.01)
    assert trajectory.values('winding.p_i')[-1] == pytest.approx(b * speed / k, rel=0.01)


def test_dc_motor_records_its_observables():
    """Test that the catalog entry of the motor records its speed and its current."""
    trajectory = create_simulation('dc_motor').set_params(dt=0.1).prepare().run(until=20.0).trajectory_
    assert trajectory.values('speed')[-1] == pytest.approx(trajectory.values('rotor.phi_d1')[-1])
    assert trajectory.values('current')[-1] == pytest.approx(trajectory.values('winding.p_i')[-1])
    assert describe('dc_motor').field == 'engineering/mechatronics'


def test_quarter_car_settles_at_the_step_height():
    """Test that the body and the wheel settle at the step height and the energy of the springs decays."""
    trajectory = Simulation(quarter_car(height=0.05), dt=0.01).prepare().run(until=10.0).trajectory_
    assert trajectory.values('car.y1')[-1] == pytest.approx(0.05, abs=1e-4)
    assert trajectory.values('car.y2')[-1] == pytest.approx(0.05, abs=1e-4)
    settled = trajectory.times > 0.5
    suspension = 16000.0 * (trajectory.values('car.y1') - trajectory.values('car.y2')).ravel() ** 2 / 2
    tire = 160000.0 * (trajectory.values('car.y2').ravel() - 0.05) ** 2 / 2
    energy = (suspension + tire)[settled]
    assert energy[-1] < 1e-3 * energy[:100].max()
    assert np.mean(energy[-100:]) < np.mean(energy[:100])


def test_quarter_car_records_its_energy():
    """Test that the catalog entry of the quarter car records the energy of its springs."""
    energy = (
        create_simulation('quarter_car')
        .set_params(dt=0.05)
        .prepare()
        .run(until=5.0)
        .trajectory_.values('energy')
        .ravel()
    )
    assert energy[0] == 0.0
    assert energy[-1] < 1e-3 * energy.max()


def test_magnetic_levitation_follows_the_periodic_solution():
    """Test that the ball follows Trzaska's periodic solution `m / (2 cosh b) exp(-b τ e) e` from its start."""
    b, m = 1.0, 0.2
    trajectory = create_simulation('magnetic_levitation', b=b, m=m).prepare().run().trajectory_
    times, v = trajectory.times, np.ravel(trajectory.values('ball.v'))
    tau = 2 / np.pi * np.arcsin(np.sin(np.pi * times / 2))
    wave = np.sign(np.cos(np.pi * times / 2))
    exact = m / (2 * np.cosh(b)) * np.exp(-b * tau * wave) * wave
    away = np.abs(np.cos(np.pi * times / 2)) > 0.05
    np.testing.assert_allclose(v[away], exact[away], atol=1e-5)


def test_vibrating_table_settles_on_a_steady_vibration_close_to_the_linear_model():
    """Test that the table vibrates steadily at the supply frequency, with nearly the amplitude of the linear model."""
    swings = []
    for linear in (False, True):
        trajectory = create_simulation('vibrating_table', linear=linear).prepare().run().trajectory_
        times, x = trajectory.times, np.ravel(trajectory.values('shaker.x'))
        late = times > 20
        swing = [np.ptp(x[(times > start) & (times < start + 4)]) for start in (20, 28, 36)]
        assert np.ptp(swing) < 1e-3 * swing[0]
        rising = times[late][1:][(x[late][:-1] < 0) & (x[late][1:] >= 0)]
        assert np.mean(np.diff(rising)) == pytest.approx(2 * np.pi / 1.5915, rel=0.01)
        swings.append(swing[0])
    assert swings[0] == pytest.approx(swings[1], rel=0.05)
