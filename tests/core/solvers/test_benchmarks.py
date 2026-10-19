"""Benchmark the Numba backend against the NumPy backend and real time."""

import time

import numpy as np
import pytest

from skcomplex.core.base import Domain, parameter
from skcomplex.core.compile import NumbaBackend, compile_system
from skcomplex.core.model import EuclideanSpace, PairPotential, Particle, ReflectingBox, System
from skcomplex.core.runtime import RealTime, Simulation
from skcomplex.core.solvers import Gillespie, VelocityVerlet
from skcomplex.core.viz import RerunObserver
from skcomplex.models.biology.ecology import LogisticGrowth, Population
from skcomplex.models.biology.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.computer_science.automata import game_of_life
from skcomplex.models.physics.statistical_mechanics import ideal_gas


class SoftSphere(PairPotential):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    k: float = parameter(100.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


def _step_time(problem, repeats=20):
    """Return the mean time of one drift and boundary evaluation after a warm-up call."""
    x, p = problem.initial_state, problem.initial_parameters
    problem.drift(0.0, x, p)
    problem.boundary(0.0, x, p)
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
        problem.boundary(0.0, x, p)
    return (time.perf_counter() - start) / repeats


@pytest.mark.benchmark
def test_numba_is_three_times_faster_on_million_member_gas():
    """Test that one step of a million-member gas takes Numba at most a third of the NumPy time."""
    gas = ideal_gas(1_000_000, random_state=0)
    numpy_time = _step_time(compile_system(gas, backend='numpy'))
    numba_time = _step_time(compile_system(gas, backend='numba'))
    assert numba_time <= numpy_time / 3


@pytest.mark.benchmark
def test_interacting_gas_of_100000_runs_in_real_time():
    """Test that 100000 soft spheres with velocity Verlet keep a real-time clock within 0.2 seconds."""
    n = 100_000
    size = float(np.sqrt(n))
    rng = np.random.default_rng(0)
    system = System(
        elements=[Particle.group('gas', n, x=rng.uniform(0, size, (n, 2)), v=rng.normal(0, 1, (n, 2)))],
        interactions=[
            SoftSphere('soft', first='gas', second='gas', cutoff=1.0),
            ReflectingBox('box', target='gas', size=size),
        ],
        space=EuclideanSpace(2),
    )
    simulation = Simulation(system, solver=VelocityVerlet(dt=0.01), dt=0.01, clock=RealTime(speed=1.0), record=False)
    simulation.run(until=2)
    assert simulation.clock_.max_lag_ <= 0.2


@pytest.mark.benchmark
def test_large_game_of_life_runs_in_real_time(tmp_path):
    """Test that a 1000 x 1000 Game of Life on Numba sustains 30 steps per second while streaming images."""
    observer = RerunObserver('life', save_path=tmp_path / 'life.rrd')
    simulation = Simulation(game_of_life((1000, 1000)), dt=1, backend='numba', record=False, observers=[observer])
    simulation.run(until=2)
    simulation.observers = [RerunObserver('life', save_path=tmp_path / 'paced.rrd')]
    simulation.clock = RealTime(30.0)
    start = time.perf_counter()
    simulation.run(until=90)
    assert 90 / (time.perf_counter() - start) >= 29.0
    assert simulation.problem_.lattice[0].fused
    assert simulation.observers[0].logged_frames_ == 91


def _epidemic():
    """Return a stochastic epidemic of 1000 individuals."""
    return System(
        elements=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0003, stochastic=True),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


def _gillespie_run(backend):
    """Return the duration and the trajectory of a compiled Gillespie run after a warm-up run."""
    simulation = Simulation(_epidemic(), backend=backend, solver=Gillespie(), dt=1.0, random_state=0)
    simulation.run(until=50)
    start = time.perf_counter()
    trajectory = simulation.run(until=50).trajectory_
    return time.perf_counter() - start, trajectory


@pytest.mark.benchmark
def test_compiled_gillespie_runs_serial_kernels():
    """Test that the default Numba backend runs a compiled Gillespie epidemic as fast as a serial one."""
    default, first = _gillespie_run(NumbaBackend())
    serial, second = _gillespie_run(NumbaBackend(parallel=False))
    assert np.array_equal(first.states, second.states)
    assert default <= 1.5 * serial + 0.005


@pytest.mark.benchmark
def test_small_system_drift_is_fast():
    """Test that the NumPy drift of 20 logistic populations takes at most 250 microseconds and is correct."""
    system = System(
        elements=[Population(f'p{index}', n=1.0 + index) for index in range(20)],
        interactions=[LogisticGrowth(f'g{index}', target=f'p{index}', r=1.0, K=100) for index in range(20)],
    )
    problem = compile_system(system, backend='numpy')
    x, p = problem.initial_state, problem.initial_parameters
    expected = x * (1 - x / 100)
    assert problem.drift(0.0, x, p) == pytest.approx(expected)
    repeats = 500
    start = time.perf_counter()
    for _ in range(repeats):
        problem.drift(0.0, x, p)
    assert (time.perf_counter() - start) / repeats <= 250e-6
