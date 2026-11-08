"""Test the `_chemotaxis` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.physiology import HyperbolicChemotaxisSystem

from ..conftest import _run_until


def test_cells_spread_conserving_their_mass_while_the_chemical_accumulates():
    """Test that the cell mass `2 n0` is conserved, the bumps spread, and the chemical grows where the cells are."""
    trajectory = _run_until(HyperbolicChemotaxisSystem(30), 0.006, dt=0.002, solver=ScipyODESolver(method='BDF'))
    cells, chemical = trajectory.values('tissue.n'), trajectory.values('tissue.c')
    np.testing.assert_allclose(cells.sum(axis=1) * (0.8 / 30) ** 2, 0.5, rtol=1e-6)
    assert np.all(np.diff(cells.max(axis=1)) < 0)
    assert np.all(np.diff(chemical.sum(axis=1)) > 0)
    assert np.corrcoef(cells[-1], chemical[-1])[0, 1] > 0.5
