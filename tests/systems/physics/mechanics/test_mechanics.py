"""Test the `_mechanics` module of `skcomplex.systems.physics.mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import symbolic
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system
from skcomplex.systems.physics.mechanics import circular_billiard, newtons_cradle


def test_pendulum_equation_is_newtons_law():
    """Test that the pendulum is compiled from Newton's law into one second-order equation."""
    equations = symbolic(create_system('pendulum')).equations
    assert list(equations) == ['bob.theta']
    assert 'sin(theta_bob(t))' in str(equations['bob.theta'].rhs())


def test_newtons_cradle_passes_momentum():
    """Test that the cradle conserves momentum through collisions and moves the last ball."""
    trajectory = Simulation(newtons_cradle(n_balls=3), dt=0.01).prepare().run(until=0.2).trajectory_
    velocities = trajectory.states[:, 1::2]
    assert [event.label for event in trajectory.events][:2] == ['collision_0', 'collision_1']
    assert velocities[-1, -1] > 0.4
    assert velocities[-1, :-1] == pytest.approx([0.0, 0.0], abs=0.02)


def test_circular_billiard_keeps_speed_and_angle():
    """Test that the billiard keeps its speed and its angle of incidence at every bounce."""
    trajectory = Simulation(circular_billiard(), dt=0.05).prepare().run(until=20).trajectory_
    speeds = np.hypot(trajectory.values('particle.vx'), trajectory.values('particle.vy'))
    assert speeds == pytest.approx(np.full(speeds.size, math.hypot(0.4, 1.0)))
    assert np.hypot(trajectory.values('particle.x'), trajectory.values('particle.y')).max() <= 1.0 + 1e-9
    incidence = [
        abs(event.before[0] * event.before[2] + event.before[1] * event.before[3]) for event in trajectory.events
    ]
    assert len(incidence) > 5
    assert np.ptp(incidence) < 1e-6
