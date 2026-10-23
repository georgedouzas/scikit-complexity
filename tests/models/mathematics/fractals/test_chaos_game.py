"""Test the `_chaos_game` module of `skcomplex.models.mathematics.fractals`."""

import numpy as np
import pytest

from skcomplex.core.analysis import box_counting_dimension
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create

from ..conftest import _final


def test_chaos_game_draws_the_sierpinski_triangle():
    """Test that the box-counting dimension of the final points is `log 3 / log 2` within 0.05."""
    points = _final('chaos_game', 30, n_points=100000).trajectory_['points.p'][-1]
    cells = np.clip((points * 128).astype(int), 0, 127)
    grid = np.zeros((128, 128), dtype=bool)
    grid[cells[:, 1], cells[:, 0]] = True
    assert box_counting_dimension(grid, [1, 2, 4, 8, 16, 32]) == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_chaos_game_agrees_across_backends():
    """Test that the NumPy and Numba backends give equal trajectories of the chaos game from one seed."""
    runs = [
        Simulation(create('chaos_game', n_points=20), dt=1, random_state=0, backend=backend).run(1000).trajectory_
        for backend in ('numpy', 'numba')
    ]
    assert np.allclose(runs[0]['points.p'], runs[1]['points.p'], atol=1e-12)


def test_chaos_game_rejects_unknown_presets():
    """Test that an unknown preset raises an error that names the presets."""
    with pytest.raises(ValidationError, match='sierpinski, barnsley'):
        create('chaos_game', preset='dragon')
