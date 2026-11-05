"""Test the `_contour_uptake` module of `skcomplex.models.biology.physiology`."""

import numpy as np

from skcomplex.core.elements import GroupElement
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.physiology import ContourCellElement, ContourUptakeInteraction


def test_contour_uptake_interaction_mixes_logistic_map_and_next_cell():
    """Test that every cell mixes its logistic map with the power of the next cell, the last with the first."""
    x, r, c, p = np.array([0.2, 0.5, 0.7]), 3.0, np.array([0.1, 0.2, 0.3]), np.array([0.5, 1.0, 2.0])
    cells = GroupElement('cells', element=ContourCellElement, size=3, values={'x': x, 'r': r, 'c': c, 'p': p})
    uptake = ContourUptakeInteraction('uptake', target='cells')
    system = System(elements=[cells], interactions=[uptake], time=DiscreteTime())
    found = Simulation(system).prepare().run(until=1).trajectory_.values('cells.x')[-1]
    expected = (1 - c) * r * x * (1 - x) + c * np.roll(x, -1) ** p
    np.testing.assert_allclose(found, expected)
