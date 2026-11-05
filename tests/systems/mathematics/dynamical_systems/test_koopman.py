"""Test the `_koopman` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.mathematics.dynamical_systems import (
    koopman_slow_manifold,
)


def test_koopman_slow_manifold_eigenfunctions_decay_exponentially():
    """Test that the Koopman eigenfunctions x1 and x2 - b x1^2 decay as exp(mu t) and exp(lam t)."""
    mu, lam = -0.05, -1.0
    trajectory = (
        Simulation(koopman_slow_manifold(1.0, -1.0, mu=mu, lam=lam), dt=0.1).prepare().run(until=40.0).trajectory_
    )
    times = 0.1 * np.arange(len(trajectory.values('flow.x1')))
    b = lam / (lam - 2 * mu)
    fast = trajectory.values('flow.x2') - b * trajectory.values('flow.x1') ** 2
    np.testing.assert_allclose(trajectory.values('flow.x1'), np.exp(mu * times), rtol=1e-5)
    np.testing.assert_allclose(fast, (-1.0 - b) * np.exp(lam * times), atol=1e-5)
