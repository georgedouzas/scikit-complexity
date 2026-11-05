"""Test the `_gyres` module of `skcomplex.systems.physics.oceanography`."""

import numpy as np
import scipy.fft

from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.oceanography import double_gyre


def test_double_gyre_forms_western_intensified_gyres_in_sverdrup_balance():
    """Test that the wind spins up two gyres of opposite sign with western extrema and a Sverdrup interior."""
    trajectory = create_simulation('double_gyre').prepare().run().trajectory_
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


def test_double_gyre_inverts_its_vorticity_once_per_rate(monkeypatch):
    """Test that a rate of the gyres solves for the streamfunction once, however many stencils read it."""
    solves = []
    transform = scipy.fft.dstn
    monkeypatch.setattr(scipy.fft, 'dstn', lambda *args, **kwargs: solves.append(1) or transform(*args, **kwargs))
    problem = compile_system(double_gyre(16), backend=NumpyBackend())
    problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert len(solves) == 1
