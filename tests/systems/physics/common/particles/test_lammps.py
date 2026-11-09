"""Test the `_lammps` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import LammpsSolver, VelocityVerletSolver
from skcomplex.systems.physics.common.particles import (
    LAMMPS_FIXES,
    LAMMPS_PAIRS,
    ConstantForceInteraction,
    DampingInteraction,
    MassPointElement,
)


def test_lammps_fixes_match_the_native_force_and_damping():
    """Test that the native LAMMPS fixes of a constant force and a damping agree with the native solver."""
    system = CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=2, values={'x': [[0.0, 0.0], [5.0, 0.0]]})],
        interactions=[
            ConstantForceInteraction('push', target='gas', F=[0.5, -0.2]),
            DampingInteraction('air', target='gas', gamma=0.3),
        ],
        space=ContinuousSpace(n_dim=2),
    )
    trajectories = [
        simulate(system, until=1, solver=solver, dt=0.1).values('gas.x_d1')
        for solver in (VelocityVerletSolver(dt=0.001), LammpsSolver(dt=0.001, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES))
    ]
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1e-3)
    assert trajectories[1][-1, 0, 0] == pytest.approx(0.5 / 0.3 * (1 - np.exp(-0.3)), rel=1e-3)
