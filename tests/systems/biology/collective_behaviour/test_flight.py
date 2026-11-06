"""Test the `_flight` module of `skcomplex.systems.biology.collective_behaviour`."""

import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.biology.collective_behaviour import BoidElement, FlightInteraction


def test_flight_interaction_moves_boids_by_velocity_on_torus():
    """Test that every boid moves by its velocity in one step, wrapping around the torus."""
    values = {'vx': [1.0, -2.0], 'vy': [0.5, 0.0]}
    flock = GroupElement('flock', element=BoidElement, size=2, values=values, location=[(29.5, 1.0), (1.0, 2.0)])
    system = System(
        elements=[flock],
        interactions=[FlightInteraction('flight', target='flock')],
        space=ContinuousSpace(extent=(30.0, 30.0), boundary='periodic'),
    )
    snapshot = Simulation(system, random_state=0).prepare().run(until=1).trajectory_.agents('flock', 1)
    np.testing.assert_allclose(snapshot.locations, [[0.5, 1.5], [29.0, 2.0]])
