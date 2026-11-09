"""Test the `_lennard_jones` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import LammpsSolver, VelocityVerletSolver
from skcomplex.systems.general.elements import PointElement
from skcomplex.systems.physics.common.particles import LAMMPS_FIXES, LAMMPS_PAIRS, LennardJonesInteraction


def _pair(interaction, distance):
    """Return a system of two particles at a distance with one pair interaction."""
    return CompositeSystem(
        elements=[GroupElement('pair', element=PointElement, size=2, values={'x': [[0.0, 0.0], [distance, 0.0]]})],
        interactions=[interaction],
        space=ContinuousSpace(n_dim=2),
    )


@pytest.mark.parametrize(
    'solver',
    [VelocityVerletSolver(dt=0.01), LammpsSolver(dt=0.01, pairs=LAMMPS_PAIRS, fixes=LAMMPS_FIXES)],
    ids=['native', 'lammps'],
)
def test_lennard_jones_force_vanishes_at_minimum(solver):
    """Test that two particles at `2 ** (1 / 6)` stay at rest."""
    system = _pair(LennardJonesInteraction('lj', first='pair', second='pair', cutoff=2.5), 2 ** (1 / 6))
    velocities = simulate(system, until=1, solver=solver, dt=0.1).get_values('pair.x_d1')
    assert np.allclose(velocities, 0.0, atol=1e-10)


def test_lennard_jones_drift_vanishes_at_minimum():
    """Test that the compiled Lennard-Jones force is zero at `2 ** (1 / 6)`."""
    system = _pair(LennardJonesInteraction('lj', first='pair', second='pair', cutoff=2.5), 2 ** (1 / 6))
    problem = PassCompiler().prepare(system).run().problem_
    assert np.allclose(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 0.0, atol=1e-12)
