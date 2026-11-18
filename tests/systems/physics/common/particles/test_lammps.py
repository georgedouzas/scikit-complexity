"""Test the `_lammps` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import LammpsSolver, VelocityVerletSolver
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import (
    LAMMPS_FIXES,
    LAMMPS_PAIRS,
    ConstantForceInteraction,
    DampingInteraction,
    MassPointElement,
)


@pytest.mark.parametrize(
    'solver',
    [VelocityVerletSolver(dt=0.001), LammpsSolver(dt=0.001, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_lammps_fixes_match_the_force_and_damping(solver):
    """Test that a constant force and a damping give the velocity `F / gamma (1 - exp(-gamma t))`."""
    system = System(
        elements=[MassPointElement('gas', x=np.array([[0.0, 0.0], [5.0, 0.0]]), size=2)],
        interactions=[
            ConstantForceInteraction('push', target='gas', F=[0.5, -0.2]),
            DampingInteraction('air', target='gas', gamma=0.3),
        ],
        space=ContinuousSpace(n_dim=2),
        solver=solver,
    )
    velocities = system.prepare().simulate(until=1, dt=0.1).get_values("gas.x'")
    assert velocities[-1, 0, 0] == pytest.approx(0.5 / 0.3 * (1 - np.exp(-0.3)), rel=1e-3)
    assert velocities[-1, 1, 1] == pytest.approx(-0.2 / 0.3 * (1 - np.exp(-0.3)), rel=1e-3)
