"""Test the `_cardiac` module of `skcomplex.models.biology.physiology`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.models.biology.physiology import (
    aliev_panfilov,
)


def test_aliev_panfilov_sustains_a_spiral():
    """Test that the broken wave in the Aliev-Panfilov tissue keeps the tissue active, as a re-entrant spiral."""
    simulation = Simulation(aliev_panfilov((50, 50)), dt=50.0, solver=RK4(dt=0.05)).run(until=400)
    excited = (simulation.trajectory_['tissue.u'] > 0.5).mean(axis=1)
    assert np.all(excited[1:] > 0.05)
