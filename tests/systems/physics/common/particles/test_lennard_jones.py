"""Test the `_lennard_jones` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.engine.solvers import LammpsSolver, VelocityVerletSolver
from skcomplex.core.systems import System
from skcomplex.systems.general.elements import PointElement
from skcomplex.systems.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS, LennardJonesInteraction


def _pair(distance, solver='auto'):
    """Return a system of two particles at a distance under the Lennard-Jones potential."""
    return System(
        elements=[PointElement('pair', x=np.array([[0.0, 0.0], [distance, 0.0]]), size=2)],
        interactions=[LennardJonesInteraction('lj', target='pair', cutoff=2.5)],
        space=ContinuousSpace(n_dim=2),
        solver=solver,
    )


@pytest.mark.parametrize(
    'solver',
    [VelocityVerletSolver(dt=0.01), LammpsSolver(dt=0.01, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_lennard_jones_force_vanishes_at_minimum(solver):
    """Test that two particles at `2 ** (1 / 6)` stay at rest."""
    velocities = _pair(2 ** (1 / 6), solver).prepare().simulate(until=1, dt=0.1).get_values("pair.x'")
    assert np.allclose(velocities, 0.0, atol=1e-10)


def test_lennard_jones_pulls_particles_beyond_the_minimum_together():
    """Test that two particles farther apart than the minimum move towards each other."""
    positions = _pair(1.5).prepare().simulate(until=0.5, dt=0.5).get_values('pair.x')[-1]
    assert positions[1, 0] - positions[0, 0] < 1.5
