"""Test the `_drivers` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.engineering.traffic import (
    CharbonneauTrafficSystem,
    RingRoadDriversSystem,
)


def test_charbonneau_cars_keep_order_and_distance():
    """Test that cars never pass each other or come closer than one unit, and that random braking stops some cars."""
    trajectory = (
        Simulation(CharbonneauTrafficSystem(50).prepare().system_, random_state=0).prepare().run(until=100).trajectory_
    )
    snapshot = trajectory.agents('cars', 100)
    order = np.argsort(snapshot.values['rank'])
    assert np.diff(snapshot.values['x'][order]).min() >= 1.0 - 1e-9
    assert snapshot.values['v'].min() < snapshot.values['v'].max()


@pytest.mark.parametrize(('n_cars', 'eps', 'free'), [(20, 0.0, True), (30, 0.02, False)])
def test_ring_road_drivers_jam_with_errors(n_cars, eps, free):
    """Test that sparse drivers without errors reach the speed limit, and errors of speed jam a denser road."""
    trajectory = (
        Simulation(RingRoadDriversSystem(n_cars, eps=eps).prepare().system_, random_state=0)
        .prepare()
        .run(until=100)
        .trajectory_
    )
    speeds = trajectory.agents('drivers', 100).values['speed']
    assert (speeds.min() == pytest.approx(40.0)) == free


def test_ring_road_drivers_read_the_road_length_from_the_space():
    """Test that drivers keep no road length of their own and stay on the ring of the space's width."""
    system = RingRoadDriversSystem(10, length=500.0).prepare().system_
    assert 'road_length' not in system.elements[0].element._get_param_names()
    trajectory = Simulation(system, random_state=0).prepare().run(until=100).trajectory_
    positions = np.concatenate([trajectory.agents('drivers', frame).locations[:, 0] for frame in range(101)])
    assert positions.min() >= 0.0
    assert positions.max() < 500.0
