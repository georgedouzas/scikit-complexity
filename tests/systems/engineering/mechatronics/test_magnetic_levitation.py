"""Test the `_magnetic_levitation` module of `skcomplex.systems.engineering.mechatronics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.mechatronics import MagneticLevitationSystem


def test_magnetic_levitation_follows_the_periodic_solution():
    """Test that the ball follows Trzaska's periodic solution `m / (2 cosh b) exp(-b τ e) e` from its start."""
    b, m = 1.0, 0.2
    trajectory = simulate(MagneticLevitationSystem(b=b, m=m))
    times, v = trajectory.times, np.ravel(trajectory.get_values('ball.v'))
    tau = 2 / np.pi * np.arcsin(np.sin(np.pi * times / 2))
    wave = np.sign(np.cos(np.pi * times / 2))
    exact = m / (2 * np.cosh(b)) * np.exp(-b * tau * wave) * wave
    away = np.abs(np.cos(np.pi * times / 2)) > 0.05
    np.testing.assert_allclose(v[away], exact[away], atol=1e-5)
