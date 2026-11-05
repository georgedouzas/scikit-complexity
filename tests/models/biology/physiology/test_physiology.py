"""Test the `_physiology` module of `skcomplex.models.biology.physiology`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.biology.physiology import fitzhugh_nagumo


def test_spiral_keeps_medium_active():
    """Test that the broken wave of the FitzHugh-Nagumo medium becomes a spiral that keeps exciting a probe."""
    trajectory = (
        Simulation(fitzhugh_nagumo((100, 100)), dt=5, solver=RK4Solver(dt=0.05)).prepare().run(until=400).trajectory_
    )
    potential = trajectory.lattice('medium.u')
    probe = potential[:, 25, 75]
    excitations = np.sum((probe[1:] > 0) & (probe[:-1] <= 0))
    assert potential[-1].max() > 1
    assert excitations >= 3
