"""Test the `_soft_sphere` module of `skcomplex.models.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import LammpsSolver, VelocityVerletSolver
from skcomplex.models.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS, SoftSphereInteraction


def _pair(interaction, distance):
    """Return a system of two particles at a distance with one pair interaction."""
    return System(
        elements=[GroupElement('pair', element=ParticleElement, size=2, values={'x': [[0.0, 0.0], [distance, 0.0]]})],
        interactions=[interaction],
        space=ContinuousSpace(n_dim=2),
    )


@pytest.mark.parametrize(
    'solver',
    [VelocityVerletSolver(dt=0.001), LammpsSolver(dt=0.001, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_soft_spheres_repel_inside_contact(solver):
    """Test that overlapping soft spheres move apart and stop interacting beyond contact."""
    system = _pair(SoftSphereInteraction('soft', first='pair', second='pair', cutoff=1.0, k=50.0), 0.8)
    trajectory = Simulation(system, solver=solver, dt=0.1).check().prepare().run(until=1).trajectory_
    separation = np.linalg.norm(trajectory.values('pair.x')[:, 1] - trajectory.values('pair.x')[:, 0], axis=1)
    assert separation[-1] > 1.0
    assert np.allclose(np.diff(separation[-3:]), np.diff(separation[-3:])[0])
