"""Test the particle solver of encounter laws against the gain and loss equations."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, SolverError, parameter, role, state
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element, Group
from skcomplex.core.experiments import Experiment
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Destruction, Encounter, Proliferation, Rate
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ParticleKinetics
from skcomplex.core.space import ContinuousTime, LatticeSpace
from skcomplex.core.system import System

CLASSES = 9
HORIZON = 2.0
PARTICLES = (1_000, 10_000, 100_000)


class Density(Element):
    """Density of particles at one activity value."""

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


def _table():
    """Return the transition table where a candidate moves one class toward the field particle with probability 1/2."""
    table = np.zeros((CLASSES, CLASSES, CLASSES))
    for candidate in range(CLASSES):
        for field in range(CLASSES):
            table[candidate, field, candidate + int(np.sign(field - candidate))] += 0.5
            table[candidate, field, candidate] += 0.5
    return table


class Exchange(Interaction):
    """Conservative encounters of a population with itself."""

    population: str = role()
    eta: float = parameter(default=1.0)

    def laws(self, s):
        """Return the encounter."""
        return [Encounter(s.population.f, s.population.f, s.eta, _table())]


class Turnover(Interaction):
    """Proliferation and destruction of a population by encounters with a fixed field."""

    population: str = role()
    field: str = role()

    def laws(self, s):
        """Return the proliferation and the destruction."""
        return [Proliferation(s.population.f, s.field.f, 0.6), Destruction(s.population.f, s.field.f, 0.2)]


class Decay(Interaction):
    """Decay of a density written as a rate law."""

    population: str = role()

    def laws(self, s):
        """Return the rate."""
        return [Rate(s.population.f, -s.population.f)]


def _profile():
    """Return a normalised initial profile peaked at a low activity."""
    grid = np.linspace(0.0, 1.0, CLASSES)
    profile = np.exp(-((grid - 0.3) ** 2) / 0.02)
    return profile / profile.sum()


def _system(*others):
    """Return a wealth distribution over 9 classes with conservative encounters and other interactions."""
    return System(
        components=[Group('wealth', element=Density, on='sites', values={'f': _profile()})],
        interactions=[Exchange('exchange', population='wealth'), *others],
        space=LatticeSpace(CLASSES, boundary='open'),
        time=ContinuousTime(),
    )


def _ode_solution(system):
    """Return the densities of the gain and loss equations at the horizon."""
    return Simulation(system, dt=HORIZON).run(until=HORIZON).trajectory_['wealth.f'][-1]


def _mean_error(system, particles, solution, replicates):
    """Return the mean L1 error of the empirical densities against the ODE solution over an ensemble."""

    def error(trajectory):
        """Return the L1 error of the last frame."""
        return float(np.abs(trajectory['wealth.f'][-1] - solution).sum())

    experiment = Experiment(
        system,
        replicates=replicates,
        until=HORIZON,
        dt=HORIZON,
        solver=ParticleKinetics(n=particles, dt=0.002),
        outputs=[error],
        random_state=0,
    )
    return float(np.mean(experiment.run().column('error')))


def test_particles_converge_to_the_ode_solution():
    """Test that the mean error of the particles falls roughly as one over the square root of their number."""
    system = _system()
    solution = _ode_solution(system)
    errors = [_mean_error(system, particles, solution, 16) for particles in PARTICLES]
    slope = np.polyfit(np.log(PARTICLES), np.log(errors), 1)[0]
    assert -0.65 < slope < -0.35
    assert errors[0] > errors[1] > errors[2]


def test_particles_keep_the_density_of_conservative_encounters():
    """Test that conservative encounters keep the number of particles, so the total density stays exact."""
    trajectory = Simulation(_system(), dt=0.5, solver=ParticleKinetics(n=2000), random_state=1).run(until=2).trajectory_
    totals = np.asarray(trajectory['wealth.f']).sum(axis=1)
    np.testing.assert_allclose(totals, totals[0], rtol=1e-12)
    assert np.all(np.asarray(trajectory['wealth.f']) >= 0)


def test_particles_sample_proliferation_and_destruction():
    """Test that proliferation and destruction change the particle density at the net rate of the equations."""
    system = System(
        components=[
            Group('wealth', element=Density, on='sites', values={'f': _profile()}),
            Group('field', element=Density, on='sites', values={'f': np.full(CLASSES, 1 / CLASSES)}),
        ],
        interactions=[Turnover('turnover', population='wealth', field='field')],
        space=LatticeSpace(CLASSES, boundary='open'),
        time=ContinuousTime(),
    )
    simulation = Simulation(system, dt=1.0, solver=ParticleKinetics(n=50_000), random_state=2).run(until=1)
    total = float(np.sum(simulation.trajectory_['wealth.f'][-1]))
    assert total == pytest.approx(np.exp(0.4), rel=0.02)


def test_particle_solver_rejects_rate_laws_in_validation():
    """Test that validation reports a rate law with the particle solver and lists the solvers that run it."""
    system = _system(Decay('decay', population='wealth'))
    report = validate(system, solver='particles')
    issue = next(item for item in report.errors if item.code == 'solver-unsupported')
    assert 'ParticleKinetics does not support the laws rate' in issue.message
    assert 'rk4' in issue.fix
    assert 'scipy' in issue.fix
    assert validate(_system(), solver='particles').ok


def test_particle_solver_rejects_rate_laws_when_run():
    """Test that the particle solver refuses a compiled problem with rate laws."""
    problem = compile_system(_system(Decay('decay', population='wealth')))
    with pytest.raises(SolverError, match='encounter laws alone'):
        ParticleKinetics().initialize(problem, np.random.default_rng(0))
