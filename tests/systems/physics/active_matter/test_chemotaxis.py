"""Test the `_chemotaxis` module of `skcomplex.systems.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import SolverError
from skcomplex.systems.physics.active_matter import KellerSegelSystem


def test_keller_segel_collapses_above_the_chemotactic_threshold():
    """Test that cells spread out below the threshold, conserving their number, and collapse in finite time above it."""
    trajectory = simulate(KellerSegelSystem((32, 32), length=32.0, chi=0.5), until=20, dt=5.0)
    rho = trajectory.get_lattice('colony.rho')
    np.testing.assert_allclose(rho.sum(axis=(1, 2)), rho[0].sum())
    assert np.ptp(rho[-1]) < 0.1 * np.ptp(rho[0])
    with pytest.raises(SolverError, match='blows up'):
        simulate(KellerSegelSystem((32, 32), length=32.0, chi=5.0), until=20, dt=5.0)
