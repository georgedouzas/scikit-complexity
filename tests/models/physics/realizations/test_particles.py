"""Test the `_particles` module of `skcomplex.models.physics.realizations`."""

import numpy as np
import pytest

from skcomplex.core.space import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyama, ScipyODE, VelocityVerlet
from skcomplex.formalisms.particles import LammpsBackend, Particle
from skcomplex.models.physics.realizations import (
    ConstantForce,
    Gravity,
    LangevinThermostat,
    LennardJones,
    LinearDrag,
    SoftSphere,
    Spring,
)
from skcomplex.models.physics.statistical_mechanics import ideal_gas


def _pair(interaction, distance):
    """Return a system of two particles at a distance with one pair interaction."""
    return System(
        components=[Particle.group('pair', 2, x=[[0.0, 0.0], [distance, 0.0]])],
        interactions=[interaction],
        space=ContinuousSpace(n_dim=2),
    )


def test_particle_constant_acceleration():
    """Test that a constant force displaces a particle by `F * t**2 / (2 * m)`."""
    system = System(
        components=[Particle('p', m=2.0)],
        interactions=[ConstantForce('push', target='p', F=[4.0, 0.0, -2.0])],
        space=ContinuousSpace(n_dim=3),
    )
    position = Simulation(system, dt=0.5).run(until=2).trajectory_['p.x'][-1]
    assert np.allclose(position, [4.0, 0.0, -2.0])


def test_spring_conserves_energy():
    """Test that two particles joined by a spring keep their total energy."""
    system = System(
        components=[Particle('a', x=[0.0, 0.0], v=[0.0, 1.0]), Particle('b', x=[2.0, 0.0], m=2.0)],
        interactions=[Spring('spring', first='a', second='b', k=3.0, length=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODE(rtol=1e-10, atol=1e-12), dt=0.1).run(until=5).trajectory_
    kinetic = 0.5 * (trajectory['a.v'] ** 2).sum(axis=1) + 0.5 * 2.0 * (trajectory['b.v'] ** 2).sum(axis=1)
    distance = np.linalg.norm(trajectory['a.x'] - trajectory['b.x'], axis=1)
    energy = kinetic + 0.5 * 3.0 * (distance - 1.0) ** 2
    assert np.allclose(energy, energy[0], rtol=1e-6)


def test_gravity_keeps_circular_orbit():
    """Test that a light body on a circular orbit keeps its radius."""
    system = System(
        components=[Particle('sun', m=1000.0), Particle('planet', x=[10.0, 0.0], v=[0.0, 10.0], m=1e-6)],
        interactions=[Gravity('gravity', first='sun', second='planet', G=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODE(rtol=1e-10, atol=1e-12), dt=0.5).run(until=6).trajectory_
    radius = np.linalg.norm(trajectory['planet.x'] - trajectory['sun.x'], axis=1)
    assert np.allclose(radius, 10.0, rtol=1e-4)


def test_linear_drag_slows_particle():
    """Test that drag decays the velocity exponentially."""
    system = System(
        components=[Particle('p', v=1.0)],
        interactions=[LinearDrag('air', target='p', gamma=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    velocity = Simulation(system, dt=0.5).run(until=2).trajectory_['p.v'][-1, 0]
    assert velocity == pytest.approx(np.exp(-1.0), rel=1e-4)


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
