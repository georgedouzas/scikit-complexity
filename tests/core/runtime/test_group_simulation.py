"""Test simulations of element groups."""

import time

import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.model import ConstantForce, EuclideanSpace, Particle, System
from skcomplex.core.runtime import RealTime, Simulation
from skcomplex.models.physics.statistical_mechanics import ideal_gas


def _pushed_gas(n=500):
    """Return a group pushed by a constant force, whose velocity change depends on the mass."""
    return System(
        components=[Particle.group('gas', n)],
        interactions=[ConstantForce('push', target='gas', F=[1.0, 0.0])],
        space=EuclideanSpace(2),
    )


def test_trajectory_returns_group_arrays():
    """Test that a group state over 100 frames has the shape `(100, 500, 2)`."""
    trajectory = Simulation(_pushed_gas(), dt=0.01).run(until=0.99).trajectory_
    assert trajectory['gas.x'].shape == (100, 500, 2)
    assert trajectory['gas.m'].shape == (100, 500)


def test_group_parameter_changes_while_running():
    """Test that setting the mass of every member applies to the frames after the change."""
    simulation = Simulation(_pushed_gas(), dt=0.01, clock=RealTime(speed=1.0)).start(until=0.4)
    time.sleep(0.15)
    simulation.update_parameters({'gas.m': 2.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    changed = int(np.argmax(np.all(trajectory['gas.m'] == 2.0, axis=1)))
    assert changed > 0
    accelerations = np.diff(trajectory['gas.v'][:, :, 0], axis=0) / 0.01
    assert np.allclose(accelerations[changed:], 0.5)
    assert np.allclose(accelerations[: changed - 1], 1.0)


def test_group_parameter_takes_per_member_values():
    """Test that a group parameter accepts one value per member."""
    simulation = Simulation(_pushed_gas(3), dt=0.1).run(until=0.1)
    simulation.update_parameters({'gas.m': [1.0, 2.0, 4.0]})
    assert simulation.problem_.parameters.read(simulation.parameters_, 'gas.m').tolist() == [1.0, 2.0, 4.0]


def test_group_parameter_with_wrong_shape_raises_error():
    """Test that a group parameter rejects values that do not broadcast to the group."""
    simulation = Simulation(_pushed_gas(3), dt=0.1).run(until=0.1)
    with pytest.raises(ValidationError, match='broadcasts'):
        simulation.update_parameters({'gas.m': [1.0, 2.0]})


def test_reflecting_box_keeps_members_inside_ode_run():
    """Test that no member crosses a wall of a reflecting box in an ODE system."""
    trajectory = Simulation(ideal_gas(500, kT=4.0, random_state=1), dt=0.01).run(until=2).trajectory_
    positions = trajectory['gas.x']
    assert np.all((positions >= 0) & (positions <= 1))


@pytest.mark.benchmark
def test_ideal_gas_runs_in_real_time():
    """Test that an ideal gas of 10000 molecules keeps a real-time clock within 0.1 seconds."""
    simulation = Simulation(ideal_gas(10_000, random_state=0), dt=0.01, clock=RealTime(speed=1.0), record=False)
    simulation.run(until=5)
    assert simulation.clock_.max_lag_ <= 0.1
