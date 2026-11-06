"""Test the `_flows` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import numpy as np
import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.mathematics.dynamical_systems import (
    finite_time_blow_up,
)

from ..conftest import _run


def test_jerk_is_bounded():
    """Test that Sprott's jerk equation stays on its bounded attractor."""
    x = create_simulation('sprott_jerk').prepare().run().trajectory_.values('jerk.x')
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10


def test_finite_time_blow_up_follows_the_hyperbola():
    """Test that `x' = x²` follows `x0 / (1 - x0 t)` towards its singularity at `t = 1 / x0`."""
    assert _run(finite_time_blow_up(1.0), 0.9, dt=0.05).values('growth.x')[-1] == pytest.approx(10.0, rel=1e-4)


def test_lorenz_attractor_is_bounded_and_visits_both_lobes():
    """Test that the Lorenz system stays bounded for 50 time units and visits both lobes of its attractor."""
    trajectory = create_simulation('lorenz').prepare().run().trajectory_
    x, z = trajectory.values('lorenz.x'), trajectory.values('lorenz.z')
    assert np.abs(x).max() < 30
    assert z.max() < 60
    late = x[len(x) // 5 :]
    assert (late > 5).any()
    assert (late < -5).any()
