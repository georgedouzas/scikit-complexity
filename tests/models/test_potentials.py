"""Test the pair potentials and the thermostat of the physics module on native and LAMMPS solvers."""

import numpy as np
import pytest

from skcomplex.core.backends import LammpsBackend
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Particle
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama, VelocityVerlet
from skcomplex.core.space import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.models.physics.realizations import LangevinThermostat, LennardJones, SoftSphere
from skcomplex.models.physics.statistical_mechanics import ideal_gas


def _pair(interaction, distance):
    """Return a system of two particles at a distance with one pair interaction."""
    return System(
        components=[Particle.group('pair', 2, x=[[0.0, 0.0], [distance, 0.0]])],
        interactions=[interaction],
        space=ContinuousSpace(n_dim=2),
    )


@pytest.mark.parametrize('solver', [VelocityVerlet(dt=0.01), LammpsBackend(dt=0.01)], ids=['native', 'lammps'])
def test_lennard_jones_force_vanishes_at_minimum(solver):
    """Test that two particles at `2 ** (1 / 6)` stay at rest."""
    system = _pair(LennardJones('lj', first='pair', second='pair', cutoff=2.5), 2 ** (1 / 6))
    velocities = Simulation(system, solver=solver, dt=0.1).run(until=1).trajectory_['pair.v']
    assert np.allclose(velocities, 0.0, atol=1e-10)


def test_lennard_jones_drift_vanishes_at_minimum():
    """Test that the compiled Lennard-Jones force is zero at `2 ** (1 / 6)`."""
    problem = compile_system(_pair(LennardJones('lj', first='pair', second='pair', cutoff=2.5), 2 ** (1 / 6)))
    assert np.allclose(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 0.0, atol=1e-12)


@pytest.mark.parametrize('solver', [VelocityVerlet(dt=0.001), LammpsBackend(dt=0.001)], ids=['native', 'lammps'])
def test_soft_spheres_repel_inside_contact(solver):
    """Test that overlapping soft spheres move apart and stop interacting beyond contact."""
    system = _pair(SoftSphere('soft', first='pair', second='pair', cutoff=1.0, k=50.0), 0.8)
    trajectory = Simulation(system, solver=solver, dt=0.1).run(until=1).trajectory_
    separation = np.linalg.norm(trajectory['pair.x'][:, 1] - trajectory['pair.x'][:, 0], axis=1)
    assert separation[-1] > 1.0
    assert np.allclose(np.diff(separation[-3:]), np.diff(separation[-3:])[0])


@pytest.mark.parametrize(
    ('solver', 'dt'),
    [(EulerMaruyama(dt=0.005), 1.0), (LammpsBackend(dt=0.01), 1.0)],
    ids=['native', 'lammps'],
)
def test_langevin_thermostat_reaches_its_temperature(solver, dt):
    """Test that a gas at temperature 0.5 under a thermostat at 2 approaches 2 within 5 percent."""
    gas = ideal_gas(2000, kT=0.5, size=40.0, random_state=0)
    system = System(
        components=gas.elements,
        interactions=[*gas.interactions, LangevinThermostat('bath', target='gas', temperature=2.0, gamma=1.0)],
        space=gas.space,
    )
    trajectory = Simulation(system, solver=solver, dt=dt, random_state=0).run(until=20).trajectory_
    temperature = (trajectory['gas.v'][-5:] ** 2).sum(axis=2).mean() / 2
    assert temperature == pytest.approx(2.0, rel=0.05)
