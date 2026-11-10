"""Test the `_flocks` module of `skcomplex.systems.biology.collective_behaviour`."""

import math

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import (
    BirdSwarmSystem,
    BoidElement,
    Boids3dSystem,
    BoidsSystem,
    FlightInteraction,
    FlockingInteraction,
    HeadingInteraction,
    SpatialBoidElement,
    SpatialFlightInteraction,
)


def test_boids_align():
    """Test that boids starting with random headings become a polarized flock."""
    trajectory = simulate(BoidsSystem().prepare().system_, until=100, dt=1, random_state=0)

    def polarization(frame):
        snapshot = trajectory.get_agents('flock', frame)
        velocity = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
        return np.hypot(*(velocity / np.hypot(velocity[:, :1], velocity[:, 1:])).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(100) > 0.7


def test_boids_3d_align_in_three_dimensions():
    """Test that boids with only the alignment rule polarize and keep their spread over the depth of the box."""
    system = Boids3dSystem(50, cohesion=0.0, separation=0.0, alignment=0.1, random_state=0).prepare().system_
    trajectory = simulate(system, dt=1, random_state=0, until=200)

    def polarization(frame):
        velocity = trajectory.get_agents('flock', frame).values['velocity']
        return np.linalg.norm((velocity / np.linalg.norm(velocity, axis=1, keepdims=True)).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(200) > 0.9
    depths = trajectory.get_agents('flock', 200).locations[:, 2]
    assert trajectory.get_agents('flock', 200).locations.shape == (50, 3)
    assert np.ptp(depths) > system.space.extent[2] / 2


def test_bird_swarm_flies_together():
    """Test that the swarm reaches the speed `sqrt(alpha / beta)` and stays together."""
    trajectory = simulate(BirdSwarmSystem(), dt=0.5, until=50)
    positions, velocities = trajectory.get_values('birds.x')[-1], trajectory.get_values('birds.x_d1')[-1]
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    assert np.median(speeds) == pytest.approx(math.sqrt(2.0), rel=0.1)
    assert np.hypot(*(positions - positions.mean(axis=0)).T).max() < 10.0


def _velocities(values, locations, interactions, until=1):
    """Return the group of boids at the last frame of a run in a periodic cube of side 10."""
    flock = GroupElement('flock', element=SpatialBoidElement, size=len(locations), values=values, location=locations)
    space = ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic')
    trajectory = simulate(
        CompositeSystem(elements=[flock], interactions=interactions, space=space),
        until=until,
        random_state=0,
    )
    return trajectory.get_agents('flock', until)


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


def test_flocking_interaction_steers_by_cohesion_and_separation_with_bounded_speed():
    """Test that two boids steer toward and away from each other across the edge of the torus, at least `min_speed`."""
    values = {'vx': [0.5, 0.0], 'vy': [0.0, 0.5], 'cohesion': 0.1, 'separation': 0.5, 'alignment': 0.0}
    flock = GroupElement('flock', element=BoidElement, size=2, values=values, location=[(29.6, 1.0), (0.4, 1.0)])
    system = CompositeSystem(
        elements=[flock],
        interactions=[FlockingInteraction('flocking', target='flock')],
        space=ContinuousSpace(extent=(30.0, 30.0), boundary='periodic'),
    )
    snapshot = simulate(system, random_state=0, until=1).get_agents('flock', 1)
    found = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
    pull = (0.1 - 0.5) * np.array([0.8, 0.0])
    steered = [np.array([0.5, 0.0]) + pull, np.array([0.0, 0.5]) - pull]
    expected = [vector * max(np.hypot(*vector), 0.5) / np.hypot(*vector) for vector in steered]
    np.testing.assert_allclose(found, expected)


def test_flight_interaction_moves_boids_by_velocity_on_torus():
    """Test that every boid moves by its velocity in one step, wrapping around the torus."""
    values = {'vx': [1.0, -2.0], 'vy': [0.5, 0.0]}
    flock = GroupElement('flock', element=BoidElement, size=2, values=values, location=[(29.5, 1.0), (1.0, 2.0)])
    system = CompositeSystem(
        elements=[flock],
        interactions=[FlightInteraction('flight', target='flock')],
        space=ContinuousSpace(extent=(30.0, 30.0), boundary='periodic'),
    )
    snapshot = simulate(system, random_state=0, until=1).get_agents('flock', 1)
    np.testing.assert_allclose(snapshot.locations, [[0.5, 1.5], [29.0, 2.0]])
