"""Test the `_billiards` module of `skcomplex.models.physics.nonlinear_oscillators`."""

import numpy as np
import pytest

from skcomplex.engine.solvers import ScipyODE
from skcomplex.models.physics.nonlinear_oscillators import (
    billiards,
)

from ..conftest import _run, _series


@pytest.mark.parametrize('shape', ['sinai', 'mushroom'])
def test_billiards_keep_the_speed_and_the_particle_on_the_table(shape):
    """Test that the billiard particle keeps its speed and never leaves the table."""
    trajectory = _run(billiards(shape), 30.0, 0.05, solver=ScipyODE())
    x, y, vx, vy = _series(trajectory, 'particle.x', 'particle.y', 'particle.vx', 'particle.vy')
    assert len(trajectory.events) > 10
    assert np.ptp(vx**2 + vy**2) < 1e-12
    if shape == 'sinai':
        assert np.abs(x).max() <= 1 + 1e-6
        assert np.abs(y).max() <= 1 + 1e-6
        assert np.hypot(x, y).min() >= 0.4 - 1e-6
        return
    stem = y < 0
    assert np.hypot(x, y)[~stem].max() <= 1 + 1e-6
    assert np.abs(x[stem]).max(initial=0.0) <= 0.4 + 1e-6
    assert y.min() >= -1 - 1e-6
