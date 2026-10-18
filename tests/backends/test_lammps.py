"""Test the LAMMPS backend against the native solvers."""

import subprocess  # nosec B404: the loading test starts a new Python process
import sys
import time

import numpy as np
import pytest
from sage.all__sagemath_symbolics import arcsinh, exp

from skcomplex.backends import LammpsBackend
from skcomplex.compile import compile_system
from skcomplex.core import BackendError, Domain, parameter
from skcomplex.model import EuclideanSpace, System, where
from skcomplex.models.ecology import LogisticGrowth, Mortality, Population, Predation
from skcomplex.models.physics import (
    ConstantForce,
    HarmonicPotential,
    LangevinThermostat,
    LennardJones,
    LinearDrag,
    PairPotential,
    Particle,
    PeriodicBox,
    ReflectingBox,
    ideal_gas,
)
from skcomplex.runtime import RealTime, Simulation
from skcomplex.solvers import VelocityVerlet

CUTOFF = 2.5


class Exponential(PairPotential):
    """Exponential repulsion `eps * exp(-r / sigma)`."""

    eps: float = parameter(1.0)
    sigma: float = parameter(0.5, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return `eps * exp(-r / sigma)`."""
        return s.eps * exp(-r / s.sigma)


class Stepwise(PairPotential):
    """Pair energy with a branch, a harmonic repulsion within a contact distance."""

    contact: float = parameter(1.3, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return `(contact - r) ** 2` within the contact distance and zero elsewhere."""
        return where(r < s.contact, (s.contact - r) ** 2, 0)


class Hyperbolic(PairPotential):
    """Pair energy with the inverse hyperbolic sine, which has no Lepton form."""

    def pair_energy(self, s, r):
        """Return `arcsinh(r)`."""
        return arcsinh(r)


def _lattice(n, spacing=1.2, offset=0.6):
    """Return the positions of a square lattice of `n` points."""
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n]
    return grid * spacing + offset


def _lennard_jones_gas(n=100, interactions=(), period=None):
    """Return a Lennard-Jones gas on a lattice with random velocities."""
    velocities = np.random.default_rng(0).normal(0, 0.5, (n, 2))
    return System(
        elements=[Particle.group('gas', n, x=_lattice(n), v=velocities)],
        interactions=[LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF, period=period), *interactions],
        space=EuclideanSpace(2),
    )


def _run(system, solver, until, dt=0.05):
    """Return the trajectory of a run."""
    return Simulation(system, solver=solver, dt=dt).run(until=until).trajectory_


def test_lammps_loads_in_new_process():
    """Test that LAMMPS starts without a missing library after installation."""
    script = "from lammps import lammps; print(lammps(cmdargs=['-screen', 'none', '-log', 'none']).version())"
    command = [sys.executable, '-c', script]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603  # nosec B603
    assert int(output) > 0


def test_lammps_trajectory_has_native_names_and_shapes():
    """Test that a LAMMPS run returns the same state names and shapes as a native run."""
    native = _run(_lennard_jones_gas(), VelocityVerlet(dt=0.005), 0.2)
    external = _run(_lennard_jones_gas(), LammpsBackend(dt=0.005), 0.2)
    for name in ('gas.x', 'gas.v'):
        assert external[name].shape == native[name].shape


def test_lammps_rejects_population_model():
    """Test that a Lotka-Volterra system names the backend and every unsupported item."""
    system = System(
        elements=[Population('rabbits', n=40), Population('foxes', n=9)],
        interactions=[
            LogisticGrowth('growth', target='rabbits'),
            Predation('hunt', predator='foxes', prey='rabbits'),
            Mortality('death', target='foxes'),
        ],
    )
    with pytest.raises(BackendError, match='LAMMPS') as error:
        Simulation(system, solver=LammpsBackend()).run(until=1)
    for name in ('rabbits', 'foxes', 'growth', 'hunt', 'death'):
        assert name in str(error.value)


def test_lammps_rejects_pair_energy_without_lepton_form():
    """Test that a pair energy with a function Lepton does not know is reported as unsupported."""
    system = System(
        elements=[Particle.group('gas', 4, x=_lattice(4))],
        interactions=[Hyperbolic('hyperbolic', first='gas', second='gas', cutoff=1.0)],
        space=EuclideanSpace(2),
    )
    with pytest.raises(BackendError, match='arcsinh has no Lepton'):
        Simulation(system, solver=LammpsBackend()).run(until=0.1)


def test_lammps_rejects_groups_with_different_masses():
    """Test that a group whose members have different masses is reported as unsupported."""
    system = System(elements=[Particle.group('gas', 2, x=_lattice(2), m=[1.0, 2.0])], space=EuclideanSpace(2))
    with pytest.raises(BackendError, match='different masses'):
        Simulation(system, solver=LammpsBackend()).run(until=0.1)


def test_lammps_rejects_mixed_boxes():
    """Test that periodic and reflecting boxes in one system are reported as unsupported."""
    system = System(
        elements=[Particle.group('a', 4, x=_lattice(4)), Particle.group('b', 4, x=_lattice(4) + 2.0)],
        interactions=[PeriodicBox('wrap', target='a', size=5.0), ReflectingBox('walls', target='b', size=5.0)],
        space=EuclideanSpace(2),
    )
    with pytest.raises(BackendError, match='periodic and reflecting'):
        Simulation(system, solver=LammpsBackend()).run(until=0.1)


def test_lammps_custom_pair_energy_matches_native_forces():
    """Test that a Lepton pair energy gives the native velocity change within 1e-8 relative error."""
    system = System(
        elements=[Particle.group('gas', 100, x=_lattice(100) + np.random.default_rng(1).normal(0, 0.05, (100, 2)))],
        interactions=[Exponential('exp', first='gas', second='gas', cutoff=CUTOFF, eps=2.0)],
        space=EuclideanSpace(2),
    )
    native = _run(system, VelocityVerlet(dt=1e-4), 1e-4, dt=1e-4)['gas.v'][-1]
    external = _run(system, LammpsBackend(dt=1e-4), 1e-4, dt=1e-4)['gas.v'][-1]
    assert np.allclose(external, native, rtol=1e-8, atol=1e-8 * np.abs(native).max())


def test_lammps_conditional_pair_energy_matches_native_forces():
    """Test that a pair energy with a condition on the distance gives the native velocity change in LAMMPS."""
    system = System(
        elements=[Particle.group('gas', 100, x=_lattice(100) + np.random.default_rng(1).normal(0, 0.1, (100, 2)))],
        interactions=[Stepwise('contact', first='gas', second='gas', cutoff=CUTOFF)],
        space=EuclideanSpace(2),
    )
    native = _run(system, VelocityVerlet(dt=1e-4), 1e-4, dt=1e-4)['gas.v'][-1]
    external = _run(system, LammpsBackend(dt=1e-4), 1e-4, dt=1e-4)['gas.v'][-1]
    assert np.abs(native).max() > 0
    assert np.allclose(external, native, rtol=1e-10, atol=1e-10 * np.abs(native).max())


def test_lammps_short_run_matches_native():
    """Test that 100 Lennard-Jones atoms agree with the native velocity Verlet run within 1e-6 after 0.1."""
    native = _run(_lennard_jones_gas(), VelocityVerlet(dt=0.005), 0.1)['gas.x']
    external = _run(_lennard_jones_gas(), LammpsBackend(dt=0.005), 0.1)['gas.x']
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_periodic_box_matches_native():
    """Test that a periodic Lennard-Jones gas agrees with the native run and stays in the box."""
    size = 12.0
    system = _lennard_jones_gas(interactions=[PeriodicBox('box', target='gas', size=size)], period=size)
    native = _run(system, VelocityVerlet(dt=0.002), 0.2)['gas.x']
    external = _run(system, LammpsBackend(dt=0.002), 0.2)['gas.x']
    assert np.allclose(external, native, atol=1e-6)
    assert np.all((external >= 0) & (external < size))


def test_lammps_walls_confine_atoms_on_their_own():
    """Test that LAMMPS walls keep atoms inside a reflecting box without the framework's boundary step."""
    problem = compile_system(ideal_gas(500, kT=4.0, size=10.0, random_state=0))
    backend = LammpsBackend(dt=0.01)
    backend.initialize(problem, np.random.default_rng(0))
    state = problem.initial_state
    for step in range(20):
        state = backend.advance(step * 0.5, state, problem.initial_parameters, (step + 1) * 0.5)
        positions = problem.states.read(state, 'gas.x')
        assert np.all((positions >= 0) & (positions <= 10))


@pytest.mark.parametrize(
    'interaction',
    [
        HarmonicPotential('trap', target='gas', k=2.0, center=[6.0, 6.0]),
        ConstantForce('push', target='gas', F=[0.5, -0.2]),
        LinearDrag('air', target='gas', gamma=0.3),
    ],
    ids=['external', 'addforce', 'viscous'],
)
def test_lammps_forces_match_native(interaction):
    """Test that external forces, constant forces, and drag agree with the native run."""
    system = _lennard_jones_gas(interactions=[interaction])
    native = _run(system, VelocityVerlet(dt=0.005), 0.2)['gas.x']
    external = _run(system, LammpsBackend(dt=0.005), 0.2)['gas.x']
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_thermostat_temperature_changes_live():
    """Test that raising the thermostat temperature during a LAMMPS run heats the gas towards it."""
    gas = ideal_gas(1000, kT=1.0, size=30.0, random_state=0)
    system = System(
        elements=gas.elements,
        interactions=[*gas.interactions, LangevinThermostat('bath', target='gas', temperature=1.0, gamma=2.0)],
        space=gas.space,
    )
    simulation = Simulation(system, solver=LammpsBackend(dt=0.01), dt=0.5, clock=RealTime(speed=10.0), random_state=0)
    simulation.start(until=20)
    time.sleep(1.0)
    simulation.update_parameters({'bath.temperature': 3.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    temperature = (trajectory['gas.v'] ** 2).sum(axis=2).mean(axis=1) / 2
    changed = int(np.argmax(trajectory['bath.temperature'] == 3.0))
    assert temperature[:changed].mean() == pytest.approx(1.0, rel=0.1)
    assert temperature[-10:].mean() == pytest.approx(3.0, rel=0.1)


def test_lammps_solver_name_resolves():
    """Test that the solver name `lammps` resolves once the backends are imported."""
    simulation = Simulation(_lennard_jones_gas(), solver='lammps', dt=0.05).run(until=0.1)
    assert isinstance(simulation.stepper_, LammpsBackend)


@pytest.mark.benchmark
def test_lammps_runs_10000_atoms_in_real_time():
    """Test that 10000 Lennard-Jones atoms keep a real-time clock within 0.2 seconds on LAMMPS."""
    n = 10_000
    system = System(
        elements=[Particle.group('gas', n, x=_lattice(n), v=np.random.default_rng(0).normal(0, 1, (n, 2)))],
        interactions=[
            LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF),
            ReflectingBox('box', target='gas', size=121.0),
        ],
        space=EuclideanSpace(2),
    )
    simulation = Simulation(system, solver=LammpsBackend(dt=0.005), dt=0.05, clock=RealTime(speed=1.0), record=False)
    simulation.run(until=5)
    assert simulation.clock_.max_lag_ <= 0.2
