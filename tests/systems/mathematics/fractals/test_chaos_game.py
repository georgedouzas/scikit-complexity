"""Test the `chaos_game` module of `skcomplex.systems.mathematics.fractals`."""

import numpy as np
import pytest

from skcomplex.core.analysis import ScalingAnalysis
from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_system
from skcomplex.systems.mathematics.fractals import ChaosPointElement, RandomAffineMapInteraction

from ..conftest import _final


def test_chaos_game_draws_the_sierpinski_triangle():
    """Test that the box-counting dimension of the final points is `log 3 / log 2` within 0.05."""
    points = _final('chaos_game', 30, n_points=100000).trajectory_.values('points.p')[-1]
    cells = np.clip((points * 128).astype(int), 0, 127)
    grid = np.zeros((128, 128), dtype=bool)
    grid[cells[:, 1], cells[:, 0]] = True
    dimension = ScalingAnalysis('box-counting', scales=[1, 2, 4, 8, 16, 32]).prepare(grid).run().exponent_
    assert dimension == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_chaos_game_agrees_across_backends():
    """Test that the NumPy and Numba backends give equal trajectories of the chaos game from one seed."""
    runs = [
        (
            Simulation(create_system('chaos_game', n_points=20), dt=1, random_state=0, backend=backend)
            .prepare()
            .run(1000)
            .trajectory_
        )
        for backend in (NumpyBackend(), NumbaBackend())
    ]
    assert np.allclose(runs[0].values('points.p'), runs[1].values('points.p'), atol=1e-12)


def test_chaos_game_rejects_unknown_presets():
    """Test that an unknown preset raises an error that names the presets."""
    with pytest.raises(ValidationError, match='sierpinski, barnsley'):
        create_system('chaos_game', preset='dragon')


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
