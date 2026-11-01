"""Test the `_heading_alignment` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.core.times import DiscreteTime
from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.active_matter import HeadingAlignmentInteraction, VicsekParticleElement


@pytest.mark.parametrize('nematic', [False, True])
def test_heading_alignment_takes_the_circular_mean_within_the_unit_distance(nematic):
    """Test that without noise one step gives the circular mean of the headings within the unit distance."""
    x = np.array([[1.0, 1.0], [1.5, 1.0], [1.2, 1.6], [4.0, 4.0]])
    theta = np.array([0.1, 2.9, -0.4, 1.0])
    flock = GroupElement('flock', element=VicsekParticleElement, size=4, values={'x': x, 'theta': theta, 'eta': 0.0})
    system = System(
        elements=[flock],
        interactions=[HeadingAlignmentInteraction('alignment', target='flock', nematic=nematic)],
        space=ContinuousSpace(extent=(5.0, 5.0), boundary='periodic'),
        time=DiscreteTime(),
    )
    values = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_.values('flock.theta')[-1]
    close = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=2) <= 1.0
    side = np.where(np.cos(theta[None, :] - theta[:, None]) >= 0, 1, -1) if nematic else 1
    weights = close * side
    expected = np.arctan2((weights * np.sin(theta)).sum(1), (weights * np.cos(theta)).sum(1))
    np.testing.assert_allclose(np.angle(np.exp(1j * (values - expected))), 0.0, atol=1e-12)
