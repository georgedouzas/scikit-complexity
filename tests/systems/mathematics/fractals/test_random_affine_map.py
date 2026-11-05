"""Test the `_random_affine_map` module of `skcomplex.systems.mathematics.fractals`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.fractals import ChaosPointElement, RandomAffineMapInteraction


def test_random_affine_map_with_one_map_applies_it():
    """Test that a single contraction toward the origin halves every position at every step."""
    points = GroupElement('points', element=ChaosPointElement, size=3, values={'p': np.ones((3, 2))})
    game = RandomAffineMapInteraction('game', target='points', maps=[(0.5, 0.0, 0.0, 0.5, 0.0, 0.0)], weights=[1.0])
    system = System(elements=[points], interactions=[game], space=ContinuousSpace(n_dim=2))
    trajectory = Simulation(system, dt=1, random_state=0).prepare().run(until=2).trajectory_
    np.testing.assert_allclose(trajectory.values('points.p')[-1], 0.25)


def test_random_affine_map_rejects_unmatched_weights():
    """Test that weights that do not match the maps raise an error that counts both."""
    game = RandomAffineMapInteraction('game', target='points', maps=[(1, 0, 0, 1, 0, 0)], weights=[1.0, 2.0])
    with pytest.raises(ValidationError, match='1 maps and 2 weights'):
        game.prepare()
