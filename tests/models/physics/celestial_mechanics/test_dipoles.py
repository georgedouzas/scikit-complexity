"""Test the `_dipoles` module of `skcomplex.models.physics.celestial_mechanics`."""

import numpy as np

from skcomplex.models.physics.celestial_mechanics import (
    stormer_problem,
)

from ..conftest import _run, _series


def test_stormer_problem_conserves_speed_and_axial_momentum():
    """Test that a charge in a dipole field keeps its speed and its canonical angular momentum about the axis."""
    trajectory = _run(stormer_problem(), 20.0, 0.02)
    x, y, z, vx, vy, vz = _series(trajectory, *(f'charge.{name}' for name in ('x', 'y', 'z', 'x_d1', 'y_d1', 'z_d1')))
    radius = np.sqrt(x**2 + y**2 + z**2)
    assert np.ptp(vx**2 + vy**2 + vz**2) < 1e-8
    assert np.ptp(x * vy - y * vx + (x**2 + y**2) / radius**3) < 1e-8
    assert radius.max() < 2
