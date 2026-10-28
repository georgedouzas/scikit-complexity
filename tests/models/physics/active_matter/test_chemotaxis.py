"""Test the `_chemotaxis` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.errors import SolverError
from skcomplex.models.physics.active_matter import keller_segel


def test_keller_segel_collapses_above_the_chemotactic_threshold():
    """Test that cells spread out below the threshold, conserving their number, and collapse in finite time above it."""
    trajectory = Simulation(keller_segel((32, 32), length=32.0, chi=0.5), dt=5.0).check().run(until=20).trajectory_
    rho = trajectory.lattice('colony.rho')
    np.testing.assert_allclose(rho.sum(axis=(1, 2)), rho[0].sum())
    assert np.ptp(rho[-1]) < 0.1 * np.ptp(rho[0])
    with pytest.raises(SolverError, match='blows up'):
        Simulation(keller_segel((32, 32), length=32.0, chi=5.0), dt=5.0).check().run(until=20)
