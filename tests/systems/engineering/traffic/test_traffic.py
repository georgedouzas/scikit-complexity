"""Test the `_traffic` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np

from skcomplex.systems.engineering.traffic import NagelSchreckenbergSystem


def test_traffic_jams_lower_flow_and_move_backward():
    """Test that dense traffic flows less than near the critical density and its jams move backward."""
    flows, velocities = {}, {}
    for density in (0.15, 0.5):
        trajectory = (
            NagelSchreckenbergSystem(1000, density=density)
            .prepare()
            .system_.simulate(
                until=600,
                dt=1,
                random_state=0,
            )
        )
        velocities[density] = trajectory.get_values('road.velocity')[100:]
        flows[density] = np.where(velocities[density] >= 0, velocities[density], 0).sum(axis=1).mean() / 1000
    assert flows[0.5] < flows[0.15]
    stopped = (velocities[0.5] == 0).astype(float)
    backward = np.mean(stopped[1:] * np.roll(stopped[:-1], -1, axis=1))
    forward = np.mean(stopped[1:] * np.roll(stopped[:-1], 1, axis=1))
    assert backward > 1.5 * forward


def test_traffic_conserves_cars():
    """Test that the number of cars on the ring never changes."""
    trajectory = NagelSchreckenbergSystem(200, density=0.3).prepare().simulate(until=100, dt=1, random_state=1)
    assert np.all((trajectory.get_values('road.velocity') >= 0).sum(axis=1) == 60)


def test_driving_shows_its_rules_in_the_equations():
    """Test that the equations name the acceleration and the movement of the cars in the order they run."""
    reads = "neighbors(road.velocity, reduce='each')"
    equations = NagelSchreckenbergSystem(50).prepare().equations_
    assert equations == {'road.velocity[i]': f'accelerate({reads}), then move({reads})'}


def _road(velocities: dict[int, float]):
    """Return the run of a ring of 100 cells with no random braking and cars of given velocities at given cells."""
    velocity = np.full(100, -1.0)
    velocity[list(velocities)] = list(velocities.values())
    system = NagelSchreckenbergSystem(100, slowdown=0.0).prepare()
    return system.simulate(until=30, dt=1, initial_state={'road.velocity': velocity})


def test_traffic_queue_dissolves_as_one_jam():
    """Test that ten stopped cars bumper to bumper leave one by one as one jam of profile `10, 9, ..., 1`."""
    (jam,) = _road(dict.fromkeys(range(10), 0.0)).events
    assert (jam.label, jam.time, jam.size, jam.duration) == ('jam', 0.0, 55, 10)
    assert jam.profile == tuple(range(10, 0, -1))


def test_traffic_jam_moves_backward():
    """Test that a stopped car at cell 10 followed by a stopped car at cell 9 at the next step is one jam."""
    (jam,) = _road({9: 1.0, 10: 0.0}).events
    assert (jam.size, jam.duration, jam.area, jam.profile) == (2, 2, 2, (1, 1))
