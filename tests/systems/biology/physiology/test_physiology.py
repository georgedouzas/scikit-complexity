"""Test the `_physiology` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.biology.physiology import FitzhughNagumoSystem


def test_spiral_keeps_medium_active():
    """Test that the broken wave of the FitzHugh-Nagumo medium becomes a spiral that keeps exciting a probe."""
    trajectory = simulate(FitzhughNagumoSystem((100, 100)), dt=5, solver=RK4Solver(dt=0.05), until=400)
    potential = trajectory.lattice('medium.u')
    probe = potential[:, 25, 75]
    excitations = np.sum((probe[1:] > 0) & (probe[:-1] <= 0))
    assert potential[-1].max() > 1
    assert excitations >= 3
