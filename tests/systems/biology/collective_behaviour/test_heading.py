"""Test the `_heading` and `_spatial_flight` modules of `skcomplex.systems.biology.collective_behaviour`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.collective_behaviour import (
    HeadingInteraction,
    SpatialBoidElement,
    SpatialFlightInteraction,
)


def _velocities(values, locations, interactions, until=1):
    """Return the group of boids at the last frame of a run in a periodic cube of side 10."""
    flock = GroupElement('flock', element=SpatialBoidElement, size=len(locations), values=values, location=locations)
    space = ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')
    simulation = Simulation(System(elements=[flock], interactions=interactions, space=space), random_state=0)
    return simulation.prepare().run(until=until).trajectory_.agents('flock', until)


def test_spatial_flight_moves_boids_by_velocity_across_the_depth_boundary():
    """Test that every boid moves by its velocity in one step, wrapping around the third side."""
    values = {'velocity': np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])}
    locations = [(1.0, 1.0, 9.5), (2.0, 2.0, 2.0)]
    snapshot = _velocities(values, locations, [SpatialFlightInteraction('flight', target='flock')])
    np.testing.assert_allclose(snapshot.locations, [[1.0, 1.0, 0.5], [3.0, 2.0, 2.0]])


def test_heading_aligns_with_the_velocity_of_the_boids_in_sight():
    """Test that alignment turns a boid to its neighbour and that speeds stay within their bounds."""
    values = {
        'velocity': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        'cohesion': 0.0,
        'separation': 0.0,
        'alignment': 0.5,
    }
    locations = [(5.0, 5.0, 4.0), (5.0, 5.0, 6.0)]
    velocities = _velocities(values, locations, [HeadingInteraction('heading', target='flock')]).values['velocity']
    assert velocities[0][2] > 0.0
    assert velocities[1][0] > 0.0
    assert np.linalg.norm(velocities, axis=1).max() <= 1.0 + 1e-12
    assert np.linalg.norm(velocities, axis=1).min() >= 0.5


def test_heading_steers_towards_far_boids_and_away_from_close_ones():
    """Test that cohesion pulls a boid along the depth towards a far boid and separation pushes it from a close one."""
    far = {'velocity': np.array([[0.5, 0.0, 0.0]] * 2), 'cohesion': 0.1, 'separation': 0.0, 'alignment': 0.0}
    locations = [(5.0, 5.0, 3.0), (5.0, 5.0, 6.0)]
    velocities = _velocities(far, locations, [HeadingInteraction('heading', target='flock')]).values['velocity']
    assert velocities[0][2] == pytest.approx(-velocities[1][2], rel=0.5)
    assert min(velocities[0][2], -velocities[1][2]) > 0.0
    close = {'velocity': np.array([[0.5, 0.0, 0.0]] * 2), 'cohesion': 0.0, 'separation': 0.5, 'alignment': 0.0}
    locations = [(5.0, 5.0, 4.8), (5.0, 5.0, 5.2)]
    velocities = _velocities(close, locations, [HeadingInteraction('heading', target='flock')]).values['velocity']
    assert velocities[0][2] < 0.0
    assert velocities[1][2] > 0.0


def test_heading_keeps_a_lone_boid_on_course():
    """Test that a boid with no other in sight keeps its velocity."""
    values = {'velocity': np.array([[0.6, 0.0, 0.0]])}
    velocities = _velocities(values, [(5.0, 5.0, 5.0)], [HeadingInteraction('heading', target='flock')]).values
    np.testing.assert_allclose(velocities['velocity'], [[0.6, 0.0, 0.0]])
