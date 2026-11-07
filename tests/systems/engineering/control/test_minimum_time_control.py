"""Test the `_minimum_time_control` module of `skcomplex.systems.engineering.control`."""

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.engineering.control import (
    minimum_time_control,
)


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
