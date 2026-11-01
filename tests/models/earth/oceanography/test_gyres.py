"""Test the `_gyres` module of `skcomplex.models.earth.oceanography`."""

import numpy as np

from skcomplex.models import run


def test_double_gyre_forms_western_intensified_gyres_in_sverdrup_balance():
    """Test that the wind spins up two gyres of opposite sign with western extrema and a Sverdrup interior."""
    trajectory = run('double_gyre').trajectory_
    frames = np.asarray(trajectory.values('psi'))
    psi, earlier = frames[-1].reshape(64, 64), frames[-11].reshape(64, 64)
    positions = (np.arange(64) + 0.5) / 64
    south, north = psi[:, :32], psi[:, 32:]
    assert south.max() > 0 > north.min()
    assert np.abs(south).max() == south.max()
    assert np.abs(north).max() == -north.min()
    for half in (south, north):
        column, _ = np.unravel_index(np.argmax(np.abs(half)), half.shape)
        assert positions[column] < 0.2
    interior = (positions > 0.4) & (positions < 0.8)
    for row in (16, 48):
        sverdrup = 1e-5 * 2 * np.pi * np.sin(2 * np.pi * positions[row]) * (1 - positions)
        error = np.abs(psi[interior, row] - sverdrup[interior]).max() / np.abs(sverdrup[interior]).max()
        assert error < 0.1
    assert np.abs(psi - earlier).max() < 1e-3 * np.abs(psi).max()
