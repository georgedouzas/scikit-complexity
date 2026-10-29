"""Test the `LammpsSolver` class."""

import time
from typing import Annotated

import numpy as np
import pytest
from lammps import lammps
from sage.all__sagemath_symbolics import arcsinh, exp, oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import GroupElement, ParticleElement
from skcomplex.core.interactions import PairPotentialInteraction
from skcomplex.core.laws.utils import where
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import RealTimeClock, Simulation
from skcomplex.engine.solvers import BaseSolver, ExternalSolver, LammpsSolver, VelocityVerletSolver
from skcomplex.errors import BackendError, SolverError
from skcomplex.models.biology.ecology import (
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
    PredationInteraction,
)
from skcomplex.models.physics.common.particles import (
    LAMMPS_FIXES,
    ConstantForceInteraction,
    HarmonicPotentialInteraction,
    LangevinThermostatInteraction,
    LennardJonesInteraction,
    LinearDragInteraction,
)
from skcomplex.models.physics.statistical_mechanics import ideal_gas
from skcomplex.specs import ParameterSpec, StepSpec

CUTOFF = 2.5


class ExponentialInteraction(PairPotentialInteraction):
    """Exponential repulsion `eps * exp(-r / sigma)`."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        eps: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.eps = eps
        self.sigma = sigma

    def _get_pair_energy(self, s, r):
        """Return `eps * exp(-r / sigma)`."""
        return s.eps * exp(-r / s.sigma)


class StepwiseInteraction(PairPotentialInteraction):
    """Pair energy with a branch, a harmonic repulsion within a contact distance."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        contact: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position, period=period)
        self.contact = contact

    def _get_pair_energy(self, s, r):
        """Return `(contact - r) ** 2` within the contact distance and zero elsewhere."""
        return where(r < s.contact, (s.contact - r) ** 2, 0)


class HyperbolicInteraction(PairPotentialInteraction):
    """Pair energy with the inverse hyperbolic sine, which has no Lepton form."""

    def _get_pair_energy(self, s, r):
        """Return `arcsinh(r)`."""
        return arcsinh(r)


def _lattice(n, spacing=1.2, offset=0.6):
    """Return the positions of a square lattice of `n` points."""
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n]
    return grid * spacing + offset


def _lennard_jones_gas(n=100, interactions=(), space=None):
    """Return a Lennard-Jones gas on a lattice with random velocities, in an open plane or a given space."""
    velocities = np.random.default_rng(0).normal(0, 0.5, (n, 2))
    return System(
        elements=[GroupElement('gas', element=ParticleElement, size=n, values={'x': _lattice(n), 'v': velocities})],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF), *interactions],
        space=space or ContinuousSpace(n_dim=2),
    )


def _run(system, solver, until, dt=0.05):
    """Return the trajectory of a run."""
    return Simulation(system, solver=solver, dt=dt).check().run(until=until).trajectory_


def test_lammps_loads():
    """Test that LAMMPS starts without a missing library after installation."""
    assert lammps(cmdargs=['-screen', 'none', '-log', 'none']).version() > 0


def test_lammps_trajectory_has_native_names_and_shapes():
    """Test that a LAMMPS run returns the same state names and shapes as a native run."""
    native = _run(_lennard_jones_gas(), VelocityVerletSolver(dt=0.005), 0.2)
    external = _run(_lennard_jones_gas(), LammpsSolver(dt=0.005), 0.2)
    for name in ('gas.x', 'gas.v'):
        assert external.values(name).shape == native.values(name).shape


def test_lammps_rejects_population_model():
    """Test that a Lotka-Volterra system names the backend and every unsupported item."""
    system = System(
        elements=[PopulationElement('rabbits', n=40), PopulationElement('foxes', n=9)],
        interactions=[
            LogisticGrowthInteraction('growth', target='rabbits'),
            PredationInteraction('hunt', predator='foxes', prey='rabbits'),
            MortalityInteraction('death', target='foxes'),
        ],
    )
    with pytest.raises(BackendError, match='LAMMPS') as error:
        Simulation(system, solver=LammpsSolver()).check().run(until=1)
    for name in ('rabbits', 'foxes', 'growth', 'hunt', 'death'):
        assert name in str(error.value)


def test_lammps_rejects_pair_energy_without_lepton_form():
    """Test that a pair energy with a function Lepton does not know is reported as unsupported."""
    system = System(
        elements=[GroupElement('gas', element=ParticleElement, size=4, values={'x': _lattice(4)})],
        interactions=[HyperbolicInteraction('hyperbolic', first='gas', second='gas', cutoff=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(BackendError, match='arcsinh has no Lepton'):
        Simulation(system, solver=LammpsSolver()).check().run(until=0.1)


def test_lammps_rejects_groups_with_different_masses():
    """Test that a group whose members have different masses is reported as unsupported."""
    system = System(
        elements=[GroupElement('gas', element=ParticleElement, size=2, values={'x': _lattice(2), 'm': [1.0, 2.0]})],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(BackendError, match='different masses'):
        Simulation(system, solver=LammpsSolver()).check().run(until=0.1)


def test_lammps_custom_pair_energy_matches_native_forces():
    """Test that a Lepton pair energy gives the native velocity change within 1e-8 relative error."""
    system = System(
        elements=[
            GroupElement(
                'gas',
                element=ParticleElement,
                size=100,
                values={'x': _lattice(100) + np.random.default_rng(1).normal(0, 0.05, (100, 2))},
            ),
        ],
        interactions=[ExponentialInteraction('exp', first='gas', second='gas', cutoff=CUTOFF, eps=2.0)],
        space=ContinuousSpace(n_dim=2),
    )
    native = _run(system, VelocityVerletSolver(dt=1e-4), 1e-4, dt=1e-4).values('gas.v')[-1]
    external = _run(system, LammpsSolver(dt=1e-4), 1e-4, dt=1e-4).values('gas.v')[-1]
    assert np.allclose(external, native, rtol=1e-8, atol=1e-8 * np.abs(native).max())


def test_lammps_conditional_pair_energy_matches_native_forces():
    """Test that a pair energy with a condition on the distance gives the native velocity change in LAMMPS."""
    system = System(
        elements=[
            GroupElement(
                'gas',
                element=ParticleElement,
                size=100,
                values={'x': _lattice(100) + np.random.default_rng(1).normal(0, 0.1, (100, 2))},
            ),
        ],
        interactions=[StepwiseInteraction('contact', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    native = _run(system, VelocityVerletSolver(dt=1e-4), 1e-4, dt=1e-4).values('gas.v')[-1]
    external = _run(system, LammpsSolver(dt=1e-4), 1e-4, dt=1e-4).values('gas.v')[-1]
    assert np.abs(native).max() > 0
    assert np.allclose(external, native, rtol=1e-10, atol=1e-10 * np.abs(native).max())


def test_lammps_short_run_matches_native():
    """Test that 100 Lennard-Jones atoms agree with the native velocity Verlet run within 1e-6 after 0.1."""
    native = _run(_lennard_jones_gas(), VelocityVerletSolver(dt=0.005), 0.1).values('gas.x')
    external = _run(_lennard_jones_gas(), LammpsSolver(dt=0.005), 0.1).values('gas.x')
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_periodic_box_matches_native():
    """Test that a periodic Lennard-Jones gas agrees with the native run and stays in the box."""
    size = 12.0
    system = _lennard_jones_gas(space=ContinuousSpace(extent=(size, size), boundary='periodic'))
    native = _run(system, VelocityVerletSolver(dt=0.002), 0.2).values('gas.x')
    external = _run(system, LammpsSolver(dt=0.002), 0.2).values('gas.x')
    assert np.allclose(external, native, atol=1e-6)
    assert np.all((external >= 0) & (external < size))


def test_lammps_walls_confine_atoms_on_their_own():
    """Test that LAMMPS walls keep atoms inside a reflecting box without the framework's boundary step."""
    problem = compile_system(ideal_gas(500, kT=4.0, size=10.0, random_state=0))
    backend = LammpsSolver(dt=0.01)
    backend.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    state = problem.initial_state
    for step in range(20):
        state = backend.run(StepSpec(step * 0.5, state, problem.initial_parameters, (step + 1) * 0.5))
        positions = problem.states.read(state, 'gas.x')
        assert np.all((positions >= 0) & (positions <= 10))


@pytest.mark.parametrize(
    'interaction',
    [
        HarmonicPotentialInteraction('trap', target='gas', k=2.0, center=[6.0, 6.0]),
        ConstantForceInteraction('push', target='gas', F=[0.5, -0.2]),
        LinearDragInteraction('air', target='gas', gamma=0.3),
    ],
    ids=['external', 'addforce', 'viscous'],
)
def test_lammps_forces_match_native(interaction):
    """Test that external forces, constant forces, and drag agree with the native run."""
    system = _lennard_jones_gas(interactions=[interaction])
    native = _run(system, VelocityVerletSolver(dt=0.005), 0.2).values('gas.x')
    external = _run(system, LammpsSolver(dt=0.005), 0.2).values('gas.x')
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_thermostat_temperature_changes_live():
    """Test that raising the thermostat temperature during a LAMMPS run heats the gas towards it."""
    gas = ideal_gas(1000, kT=1.0, size=30.0, random_state=0)
    system = System(
        elements=gas.elements,
        interactions=[
            *gas.interactions,
            LangevinThermostatInteraction('bath', target='gas', temperature=1.0, gamma=2.0),
        ],
        space=gas.space,
    )
    simulation = Simulation(
        system,
        solver=LammpsSolver(dt=0.01, fixes=LAMMPS_FIXES),
        dt=0.5,
        clock=RealTimeClock(speed=10.0),
        random_state=0,
    ).check()
    simulation.start(until=20)
    time.sleep(1.0)
    simulation.update_parameters({'bath.temperature': 3.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    temperature = (trajectory.values('gas.v') ** 2).sum(axis=2).mean(axis=1) / 2
    changed = int(np.argmax(trajectory.values('bath.temperature') == 3.0))
    assert temperature[:changed].mean() == pytest.approx(1.0, rel=0.1)
    assert temperature[-10:].mean() == pytest.approx(3.0, rel=0.1)


@pytest.mark.benchmark
def test_lammps_runs_10000_atoms_in_real_time():
    """Test that 10000 Lennard-Jones atoms keep a real-time clock within 0.2 seconds on LAMMPS."""
    n = 10_000
    system = System(
        elements=[
            GroupElement(
                'gas',
                element=ParticleElement,
                size=n,
                values={'x': _lattice(n), 'v': np.random.default_rng(0).normal(0, 1, (n, 2))},
            ),
        ],
        interactions=[
            LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF),
        ],
        space=ContinuousSpace(extent=(121.0, 121.0), boundary='reflecting'),
    )
    simulation = Simulation(
        system,
        solver=LammpsSolver(dt=0.005),
        dt=0.05,
        clock=RealTimeClock(speed=1.0),
        record=False,
    ).check()
    simulation.run(until=5)
    assert simulation.clock_.max_lag_ <= 0.2


def test_lammps_solver_is_an_external_solver():
    """Test that the LAMMPS solver derives from the external solver and the solver base."""
    assert issubclass(LammpsSolver, ExternalSolver)
    assert issubclass(LammpsSolver, BaseSolver)


def test_lammps_final_run_closes_the_instance():
    """Test that the last frame of a run closes the LAMMPS instance, which runs again only once refitted."""
    problem = compile_system(_lennard_jones_gas())
    simulation = Simulation(problem, solver=LammpsSolver(dt=0.005), dt=0.05).check().run(until=0.1)
    assert simulation.stepper_.released_
    with pytest.raises(SolverError, match='closed its LAMMPS instance'):
        simulation.extend(until=0.2)
    solver = simulation.stepper_.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    assert not solver.released_
