"""Test the `LammpsSolver` class."""

import os
import time
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from lammps import lammps
from sage.all__sagemath_symbolics import SR, arcsinh, exp, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw, RateLaw
from skcomplex.core.components.laws.utils import where
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import BaseSolver, ExternalSolver, LammpsSolver, StepSpec, VelocityVerletSolver
from skcomplex.core.errors import BackendError, SolverError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.common.populations import (
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.ecology import PredationInteraction
from skcomplex.systems.physics.common.particles import (
    LAMMPS_FIXES,
    ConstantForceInteraction,
    DampingInteraction,
    LangevinThermostatInteraction,
)
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.physics.waves import HarmonicPotentialInteraction

from ....conftest import BodyElement

CUTOFF = 2.5


def _distance(s):
    """Return the distance between the two members of a pair."""
    separation = s.first.x - s.second.x
    return sqrt(separation.dot_product(separation))


class LennardJonesInteraction(Interaction):
    """Lennard-Jones pair energy, which LAMMPS runs as a Lepton pair style."""

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        cutoff: float = 1.0,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 1.0,
        sigma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label, within=RadiusNeighborhood(cutoff))
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.epsilon = epsilon
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the energy `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)` of a pair."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            r = _distance(s)
            self.laws_ = [ObjectiveLaw(4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6))]


def _exponential(eps=1.0, sigma=0.5):
    """Return the laws of the exponential repulsion `eps * exp(-r / sigma)` of a pair."""
    return lambda s: [ObjectiveLaw(eps * exp(-_distance(s) / sigma))]


def _stepwise(contact=1.3):
    """Return the laws of a harmonic repulsion within a contact distance, an energy with a branch."""

    def laws(s):
        r = _distance(s)
        return [ObjectiveLaw(where(r < contact, (contact - r) ** 2, 0))]

    return laws


def _hyperbolic(s):
    """Return the laws of the energy `arcsinh(r)`, which has no Lepton form."""
    return [ObjectiveLaw(arcsinh(_distance(s)))]


def _lattice(n, spacing=1.2, offset=0.6):
    """Return the positions of a square lattice of `x` points."""
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n]
    return grid * spacing + offset


def _lennard_jones_gas(n=100, interactions=(), space=None):
    """Return a Lennard-Jones gas on a lattice with random velocities, in an open plane or a given space."""
    velocities = np.random.default_rng(0).normal(0, 0.5, (n, 2))
    return CompositeSystem(
        elements=[BodyElement('gas', x=_lattice(n), size=n, values={"x'": velocities})],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF), *interactions],
        space=space or ContinuousSpace(n_dim=2),
    )


def _run(system, solver, until, dt=0.05):
    """Return the trajectory of a run."""
    return simulate(system, until=until, solver=solver, dt=dt)


def test_lammps_starts_mpi_without_infiniband_by_default():
    """Test that the solver keeps MPI off the InfiniBand devices, which kill it on some machines."""
    assert os.environ['UCX_TLS']
    assert os.environ['FI_PROVIDER']


def test_lammps_loads():
    """Test that LAMMPS starts without a missing library after installation."""
    assert lammps(cmdargs=['-screen', 'none', '-log', 'none']).version() > 0


def test_lammps_trajectory_has_native_names_and_shapes():
    """Test that a LAMMPS run returns the same state names and shapes as a native run."""
    native = _run(_lennard_jones_gas(), VelocityVerletSolver(dt=0.005), 0.2)
    external = _run(_lennard_jones_gas(), LammpsSolver(dt=0.005), 0.2)
    for name in ('gas.x', "gas.x'"):
        assert external.get_values(name).shape == native.get_values(name).shape


def test_lammps_rejects_population_system():
    """Test that a Lotka-Volterra system names the backend and every unsupported item."""
    system = System(
        elements=[PopulationElement('rabbits', x=40), PopulationElement('foxes', x=9)],
        interactions=[
            LogisticGrowthInteraction('growth', target='rabbits'),
            PredationInteraction('hunt', predator='foxes', prey='rabbits'),
            MortalityInteraction('death', target='foxes'),
        ],
        solver=LammpsSolver(),
    )
    with pytest.raises(BackendError, match='LAMMPS') as error:
        system.prepare().simulate(until=1)
    for name in ('rabbits', 'foxes', 'growth', 'hunt', 'death'):
        assert name in str(error.value)


def test_lammps_rejects_pair_energy_without_lepton_form():
    """Test that a pair energy with a function Lepton does not know is reported as unsupported."""
    system = CompositeSystem(
        elements=[BodyElement('gas', x=_lattice(4), size=4)],
        interactions=[Interaction('hyperbolic', among='gas', within=RadiusNeighborhood(1.0), laws=_hyperbolic)],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(BackendError, match='arcsinh has no Lepton'):
        simulate(system, until=0.1, solver=LammpsSolver())


def test_lammps_rejects_groups_with_different_masses():
    """Test that a group whose members have different masses is reported as unsupported."""
    system = CompositeSystem(
        elements=[BodyElement('gas', x=_lattice(2), m=[1.0, 2.0], size=2)],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(BackendError, match='different values of m'):
        simulate(system, until=0.1, solver=LammpsSolver())


def test_lammps_custom_pair_energy_matches_native_forces():
    """Test that a Lepton pair energy gives the native velocity change within 1e-8 relative error."""
    system = CompositeSystem(
        elements=[
            BodyElement(
                'gas',
                x=_lattice(100) + np.random.default_rng(1).normal(0, 0.05, (100, 2)),
                size=100,
            ),
        ],
        interactions=[
            Interaction('exp', among='gas', within=RadiusNeighborhood(CUTOFF), laws=_exponential(eps=2.0)),
        ],
        space=ContinuousSpace(n_dim=2),
    )
    native = _run(system, VelocityVerletSolver(dt=1e-4), 1e-4, dt=1e-4).get_values("gas.x'")[-1]
    external = _run(system, LammpsSolver(dt=1e-4), 1e-4, dt=1e-4).get_values("gas.x'")[-1]
    assert np.allclose(external, native, rtol=1e-8, atol=1e-8 * np.abs(native).max())


def test_lammps_conditional_pair_energy_matches_native_forces():
    """Test that a pair energy with a condition on the distance gives the native velocity change in LAMMPS."""
    system = CompositeSystem(
        elements=[
            BodyElement(
                'gas',
                x=_lattice(100) + np.random.default_rng(1).normal(0, 0.1, (100, 2)),
                size=100,
            ),
        ],
        interactions=[Interaction('contact', among='gas', within=RadiusNeighborhood(CUTOFF), laws=_stepwise())],
        space=ContinuousSpace(n_dim=2),
    )
    native = _run(system, VelocityVerletSolver(dt=1e-4), 1e-4, dt=1e-4).get_values("gas.x'")[-1]
    external = _run(system, LammpsSolver(dt=1e-4), 1e-4, dt=1e-4).get_values("gas.x'")[-1]
    assert np.abs(native).max() > 0
    assert np.allclose(external, native, rtol=1e-10, atol=1e-10 * np.abs(native).max())


def test_lammps_short_run_matches_native():
    """Test that 100 Lennard-Jones atoms agree with the native velocity Verlet run within 1e-6 after 0.1."""
    native = _run(_lennard_jones_gas(), VelocityVerletSolver(dt=0.005), 0.1).get_values('gas.x')
    external = _run(_lennard_jones_gas(), LammpsSolver(dt=0.005), 0.1).get_values('gas.x')
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_periodic_box_matches_native():
    """Test that a periodic Lennard-Jones gas agrees with the native run and stays in the box."""
    size = 12.0
    system = _lennard_jones_gas(space=ContinuousSpace(extent=(size, size), boundary='periodic'))
    native = _run(system, VelocityVerletSolver(dt=0.002), 0.2).get_values('gas.x')
    external = _run(system, LammpsSolver(dt=0.002), 0.2).get_values('gas.x')
    assert np.allclose(external, native, atol=1e-6)
    assert np.all((external >= 0) & (external < size))


def test_lammps_walls_confine_atoms_on_their_own():
    """Test that LAMMPS walls keep atoms inside a reflecting box without the framework's boundary step."""
    problem = (
        PassCompiler().prepare(IdealGasSystem(500, kT=4.0, size=10.0, random_state=0).prepare().system_).run().problem_
    )
    backend = LammpsSolver(dt=0.01)
    backend.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    state = problem.initial_state
    for step in range(20):
        state = backend.run(StepSpec(step * 0.5, state, problem.initial_parameters, (step + 1) * 0.5)).state_
        positions = problem.states.read(state, 'gas.x')
        assert np.all((positions >= 0) & (positions <= 10))


@pytest.mark.parametrize(
    'interaction',
    [
        HarmonicPotentialInteraction('trap', target='gas', k=2.0, center=[6.0, 6.0]),
        ConstantForceInteraction('push', target='gas', F=[0.5, -0.2]),
        DampingInteraction('air', target='gas', gamma=0.3),
    ],
    ids=['external', 'addforce', 'viscous'],
)
def test_lammps_forces_match_native(interaction):
    """Test that external forces, constant forces, and drag agree with the native run."""
    system = _lennard_jones_gas(interactions=[interaction])
    native = _run(system, VelocityVerletSolver(dt=0.005), 0.2).get_values('gas.x')
    external = _run(system, LammpsSolver(dt=0.005), 0.2).get_values('gas.x')
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_thermostat_temperature_changes_live():
    """Test that raising the thermostat temperature during a LAMMPS run heats the gas towards it."""
    gas = IdealGasSystem(1000, kT=1.0, size=30.0, random_state=0).prepare().system_
    system = System(
        elements=gas.elements,
        interactions=[
            *gas.interactions,
            LangevinThermostatInteraction('bath', target='gas', temperature=1.0, gamma=2.0),
        ],
        space=gas.space,
        solver=LammpsSolver(dt=0.01, fixes=LAMMPS_FIXES),
    )
    trajectory = system.prepare().simulate(
        until=20,
        dt=0.5,
        random_state=0,
        schedule={'bath.temperature': lambda t: 1.0 if t < 10 else 3.0},
    )
    temperature = (trajectory.get_values("gas.x'") ** 2).sum(axis=2).mean(axis=1) / 2
    changed = int(np.argmax(trajectory.get_values('bath.temperature') == 3.0))
    assert temperature[:changed].mean() == pytest.approx(1.0, rel=0.1)
    assert temperature[-10:].mean() == pytest.approx(3.0, rel=0.1)


@pytest.mark.benchmark
def test_lammps_runs_10000_atoms_in_real_time():
    """Test that 10000 Lennard-Jones atoms keep a real-time clock within 0.2 seconds on LAMMPS."""
    n = 10_000
    system = CompositeSystem(
        elements=[
            BodyElement(
                'gas',
                x=_lattice(n),
                size=n,
                values={"x'": np.random.default_rng(0).normal(0, 1, (n, 2))},
            ),
        ],
        interactions=[
            LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF),
        ],
        space=ContinuousSpace(extent=(121.0, 121.0), boundary='reflecting'),
    )
    problem = PassCompiler().prepare(system).run().problem_
    solver = LammpsSolver(dt=0.005, until=5.0).prepare(problem)
    state, lags = solver.initial_state_, []
    start = time.perf_counter()
    for frame in range(100):
        step = StepSpec(frame * 0.05, state, problem.initial_parameters, (frame + 1) * 0.05)
        state = solver.run(step, final=frame == 99).state_
        lags.append(time.perf_counter() - start - (frame + 1) * 0.05)
    assert max(lags) <= 0.2


def test_lammps_solver_is_an_external_solver():
    """Test that the LAMMPS solver derives from the external solver and the solver base."""
    assert issubclass(LammpsSolver, ExternalSolver)
    assert issubclass(LammpsSolver, BaseSolver)


def test_lammps_final_run_closes_the_instance():
    """Test that the last frame of a run closes the LAMMPS instance, which runs again only once refitted."""
    problem = PassCompiler().prepare(_lennard_jones_gas()).run().problem_
    solver = LammpsSolver(dt=0.005, until=0.1).prepare(problem)
    step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 0.1)
    state = solver.run(step, final=True).state_
    assert solver.released_
    with pytest.raises(SolverError, match='closed its LAMMPS instance'):
        solver.run(StepSpec(0.1, state, problem.initial_parameters, 0.2))
    solver = solver.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    assert not solver.released_


def test_lammps_solver_get_lepton_expression():
    """Test `LammpsSolver._get_lepton_expression` on a small case."""
    x = symbol('x')
    assert LammpsSolver._get_lepton_expression(4 * (x**-12 - x**-6), x) == '4/r^12 - 4/r^6'


class CrowdsLennardJonesInteraction(Interaction):
    """Lennard-Jones force between every pair of members of two crowds closer than the cutoff."""

    def __init__(self, label: str, *, first: Role, second: Role, epsilon: float = 1.0, sigma: float = 1.0) -> None:
        super().__init__(label, within=RadiusNeighborhood(CUTOFF))
        self.first = first
        self.second = second
        self.epsilon = epsilon
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forces `-U'(r)` of `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)` on both members."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            separation = s.first.x - s.second.x
            r = sqrt(separation.dot_product(separation))
            distance = SR.var('r', domain='positive')
            energy = 4 * s.epsilon * ((s.sigma / distance) ** 12 - (s.sigma / distance) ** 6)
            force = -energy.diff(distance).subs({distance: r}) * separation / r
            self.laws_ = [RateLaw(s.first.x.diff(s.t, 2), force), RateLaw(s.second.x.diff(s.t, 2), -force)]


def _lennard_jones_coefficients(value):
    """Return the coefficients of `lj/cut`."""
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def _crowds(solver):
    """Return the positions of two crowds of four atoms that a Lennard-Jones force within the cutoff couples."""
    space = ContinuousSpace(n_dim=2)
    left = Element('left', values={'x': _lattice(4, spacing=2.4), "x'": np.zeros((4, 2))}, size=4, space=space)
    right = Element('right', values={'x': _lattice(4, spacing=2.4) + 1.2, "x'": np.zeros((4, 2))}, size=4, space=space)
    forces = CrowdsLennardJonesInteraction('lj', first='left', second='right')
    system = System(elements=[left, right], interactions=[forces], space=space, solver=solver)
    return system.prepare().simulate(until=0.1, dt=0.05).get_values('right.x')


def test_lammps_runs_a_pair_force_within_a_radius_as_its_native_pair_style():
    """Test that a force on the pairs of two crowds within a radius runs as `lj/cut` and agrees with the native run."""
    pairs = {CrowdsLennardJonesInteraction: ('lj/cut', _lennard_jones_coefficients)}
    native = _crowds(VelocityVerletSolver(dt=0.005))
    external = _crowds(LammpsSolver(dt=0.005, pairs=pairs))
    assert np.abs(native[-1] - native[0]).max() > 1e-4
    assert np.allclose(external, native, atol=1e-6)


def test_lammps_runs_a_pair_energy_as_a_lepton_pair_style():
    """Test that the energy of the pairs of a gas becomes a Lepton pair style of the distance."""
    problem = PassCompiler().prepare(_lennard_jones_gas()).run().problem_
    solver = LammpsSolver(dt=0.005).prepare(problem)
    commands = solver._get_pair_commands(problem, problem.initial_parameters)
    assert f'pair_coeff 1 1 lepton "4.0/r^12 - 4.0/r^6" {CUTOFF}' in commands
