"""Test the `_flocking` module of `skcomplex.models.biology.collective_behaviour`."""

import numpy as np

from skcomplex.core.elements import GroupElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import BoidElement, FlockingInteraction


def test_flocking_interaction_steers_by_cohesion_and_separation_with_bounded_speed():
    """Test that two boids steer toward and away from each other across the edge of the torus, at least `min_speed`."""
    values = {'vx': [0.5, 0.0], 'vy': [0.0, 0.5], 'cohesion': 0.1, 'separation': 0.5, 'alignment': 0.0}
    flock = GroupElement('flock', element=BoidElement, size=2, values=values, location=[(29.6, 1.0), (0.4, 1.0)])
    system = System(
        elements=[flock],
        interactions=[FlockingInteraction('flocking', target='flock')],
        space=ContinuousSpace(extent=(30.0, 30.0), boundary='periodic'),
    )
    snapshot = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_.agents('flock', 1)
    found = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
    pull = (0.1 - 0.5) * np.array([0.8, 0.0])
    steered = [np.array([0.5, 0.0]) + pull, np.array([0.0, 0.5]) - pull]
    expected = [vector * max(np.hypot(*vector), 0.5) / np.hypot(*vector) for vector in steered]
    np.testing.assert_allclose(found, expected)
