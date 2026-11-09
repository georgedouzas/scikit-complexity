"""Test the `_evacuation` module of `skcomplex.systems.social.crowds`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.systems.social.crowds import EvacuationSystem


def test_evacuation_empties_the_building():
    """Test that the evacuated fraction grows monotonically to the whole crowd and the density is kept."""
    trajectory = simulate(EvacuationSystem(), until=40, dt=2.0, random_state=0)
    evacuated = trajectory.values('evacuated')
    assert evacuated[0] == 0.0
    assert np.all(np.diff(evacuated) >= -1e-12)
    assert evacuated[-1] > 0.99
    np.testing.assert_allclose(np.asarray(trajectory.values('crowd.x')).sum(axis=1), 1.0, rtol=1e-9)


def test_evacuation_runs_with_particles():
    """Test that the particle solver evacuates the building like the gain and loss equations."""
    equations = simulate(EvacuationSystem(), until=6, dt=3.0, random_state=0)
    particles = simulate(EvacuationSystem(), until=6, dt=3.0, solver=KineticSolver(n=20_000, dt=0.01), random_state=0)
    np.testing.assert_allclose(particles.values('evacuated'), equations.values('evacuated'), atol=0.03)
