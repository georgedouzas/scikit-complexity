"""Test the `_evacuation` module of `skcomplex.models.social.crowds`."""

import numpy as np

from skcomplex.formalisms.kinetics import KineticSolver

from ...conftest import _trajectory


def test_evacuation_empties_the_building():
    """Test that the evacuated fraction grows monotonically to the whole crowd and the density is kept."""
    trajectory = _trajectory('evacuation', 40, 2.0)
    evacuated = trajectory.values('evacuated')
    assert evacuated[0] == 0.0
    assert np.all(np.diff(evacuated) >= -1e-12)
    assert evacuated[-1] > 0.99
    np.testing.assert_allclose(np.asarray(trajectory.values('crowd.f')).sum(axis=1), 1.0, rtol=1e-9)


def test_evacuation_runs_with_particles():
    """Test that the particle solver evacuates the building like the gain and loss equations."""
    equations = _trajectory('evacuation', 6, 3.0)
    particles = _trajectory('evacuation', 6, 3.0, solver=KineticSolver(n=20_000, dt=0.01))
    np.testing.assert_allclose(particles.values('evacuated'), equations.values('evacuated'), atol=0.03)
