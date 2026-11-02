"""Test the `_swimming` module of `skcomplex.models.biology.collective_behaviour`."""

import math

import numpy as np

from skcomplex.core.elements import GroupElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import ShoalMemberElement, SwimmingInteraction


def test_swimming_interaction_turns_by_deviation_and_swims():
    """Test that a fish turns its heading by its deviation and swims `speed` along the new heading."""
    values = {'vx': 1.0, 'vy': 0.0, 'deviation': 0.5, 'speed': 2.0}
    shoal = GroupElement('shoal', element=ShoalMemberElement, size=1, values=values, location=[(99.0, 1.0)])
    system = System(
        elements=[shoal],
        interactions=[SwimmingInteraction('swimming', target='shoal')],
        space=ContinuousSpace(extent=(100.0, 100.0), boundary='periodic'),
    )
    snapshot = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_.agents('shoal', 1)
    heading = [math.cos(0.5), math.sin(0.5)]
    np.testing.assert_allclose([snapshot.values['vx'][0], snapshot.values['vy'][0]], heading)
    np.testing.assert_allclose(snapshot.locations[0], np.mod([99.0 + 2 * heading[0], 1.0 + 2 * heading[1]], 100.0))
