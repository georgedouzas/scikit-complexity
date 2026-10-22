"""Benchmark the Numba backend against the NumPy backend and real time, and the sparse Jacobians of stiff fields."""

import time
from dataclasses import replace

import numpy as np
import pytest

from skcomplex.core.backends import NumbaBackend
from skcomplex.core.base import Domain, parameter
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element, Field, Particle
from skcomplex.core.interactions import Interaction, PairPotential
from skcomplex.core.laws import Event, Rate, delayed, laplacian, total
from skcomplex.core.runtime import RealTime, Simulation
from skcomplex.core.solvers import CrankNicolson, Gillespie, ScipyODE, VelocityVerlet
from skcomplex.core.space import ContinuousSpace, LatticeSpace
from skcomplex.core.system import System
from skcomplex.core.viz import RerunObserver
from skcomplex.models.biology.ecology import LogisticGrowth, Population, fisher_kpp
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
        components=[Particle.group('gas', n, x=rng.uniform(0, size, (n, 2)), v=rng.normal(0, 1, (n, 2)))],
        interactions=[
            SoftSphere('soft', first='gas', second='gas', cutoff=1.0),
        ],
        space=ContinuousSpace(size, size, boundary='reflecting'),
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
        components=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
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
        components=[Population(f'p{index}', n=1.0 + index) for index in range(20)],
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


STIFF_SIZE = 4000


def _stiff_fisher_problem():
    """Return Fisher-KPP on 4000 points whose diffusion makes the largest eigenvalue about 1600."""
    return compile_system(fisher_kpp(STIFF_SIZE, D=1.0, spacing=0.05), backend='numpy')


def _counted(problem):
    """Return a problem whose drift counts its evaluations, with the counter."""
    calls = [0]
    drift = problem.drift

    def counting(t, x, p):
        calls[0] += 1
        return drift(t, x, p)

    return replace(problem, drift=counting), calls


def _timed_scipy(problem, method, jacobian, until=10.0):
    """Return the duration, the final state, the drift evaluations, and the stepper of one call of ScipyODE."""
    problem, calls = _counted(problem)
    stepper = ScipyODE(method=method, rtol=1e-6, jacobian=jacobian)
    stepper.initialize(problem, np.random.default_rng(0))
    start = time.perf_counter()
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, until)
    return time.perf_counter() - start, state, calls[0], stepper


@pytest.mark.benchmark
def test_sparse_jacobians_speed_up_stiff_fields(record_property):
    """Test that the pattern, Radau, and the exact Jacobian are far faster than BDF on the dense path."""
    problem = _stiff_fisher_problem()
    _timed_scipy(problem, 'BDF', 'auto', until=0.1)
    pattern, reference, pattern_calls, stepper = _timed_scipy(problem, 'BDF', 'auto')
    dense, dense_state, _, _ = _timed_scipy(problem, 'BDF', 'dense')
    radau, radau_state, _, _ = _timed_scipy(problem, 'Radau', 'auto')
    exact, exact_state, exact_calls, exact_stepper = _timed_scipy(problem, 'BDF', 'exact')
    split = Simulation(problem, solver=CrankNicolson(dt=0.01), dt=10.0, record=False)
    start = time.perf_counter()
    split.run(until=10.0)
    crank_nicolson = time.perf_counter() - start
    for name, value in [('pattern', pattern), ('radau', radau), ('exact', exact), ('crank_nicolson', crank_nicolson)]:
        record_property(f'dense_over_{name}', dense / value)
    assert dense >= 20 * pattern
    assert radau * 10 <= dense
    assert exact * 10 <= dense
    for state in [dense_state, radau_state, exact_state]:
        assert np.linalg.norm(state - reference) / np.linalg.norm(reference) < 1e-5
    assert pattern_calls > stepper._solver.nfev
    assert exact_calls == exact_stepper._solver.nfev


@pytest.mark.benchmark
def test_banded_lsoda_speeds_up_stiff_fields(record_property):
    """Test that LSODA with half bandwidths of 1 is at least 10 times faster than without them."""
    problem = _stiff_fisher_problem()
    banded, state, _, stepper = _timed_scipy(problem, 'LSODA', 'auto')
    dense, dense_state, _, _ = _timed_scipy(problem, 'LSODA', 'dense')
    record_property('dense_over_banded', dense / banded)
    assert stepper.jacobian_options_ == {'lband': 1, 'uband': 1}
    assert banded * 10 <= dense
    assert np.linalg.norm(state - dense_state) / np.linalg.norm(dense_state) < 1e-5


StiffThreshold = Interaction.define(
    'StiffThreshold',
    roles=['line'],
    parameters={'level': 0.0},
    laws=lambda s: [Event(total(s.line.u) - s.level, terminal=True)],
)


def _event_run(jacobian):
    """Return the duration and the event time of stiff Fisher-KPP that stops when its total grows by half."""
    line = fisher_kpp(STIFF_SIZE, D=1.0, spacing=0.05).elements[0]
    level = 1.5 * line.values['u'].sum()
    system = System(components=[line], interactions=[StiffThreshold('threshold', line='line', level=level)])
    simulation = Simulation(system, solver=ScipyODE(method='BDF', jacobian=jacobian), dt=1.0, backend='numpy')
    start = time.perf_counter()
    times = simulation.run(until=10.0).trajectory_.times
    return time.perf_counter() - start, times[-1]


@pytest.mark.benchmark
def test_sparse_jacobians_speed_up_events_on_stiff_fields(record_property):
    """Test that the pattern locates the event of a stiff field at the dense path's time, at least 10 times faster."""
    _event_run('auto')
    pattern, time_pattern = _event_run('auto')
    dense, time_dense = _event_run('dense')
    record_property('dense_over_pattern', dense / pattern)
    assert 0 < time_pattern < 10
    assert abs(time_pattern - time_dense) < 1e-6
    assert pattern * 10 <= dense


DiffusiveHutchinson = Element.define(
    'DiffusiveHutchinson',
    states={'u': 0.0},
    parameters={'D': 1.0, 'r': 1.2},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - delayed(s.u, 1.0)))],
)


def _hutchinson_run(jacobian):
    """Return the duration and the final state of a diffusive Hutchinson equation on 2000 points with BDF."""
    size = 2000
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    field = Field('line', DiffusiveHutchinson, LatticeSpace(size, spacing=0.05), values={'u': initial})
    simulation = Simulation(
        System(components=[field]),
        solver=ScipyODE(method='BDF', jacobian=jacobian),
        dt=1.0,
        backend='numpy',
    )
    start = time.perf_counter()
    state = simulation.run(until=5.0).trajectory_.states[-1]
    return time.perf_counter() - start, state


@pytest.mark.benchmark
def test_sparse_jacobians_keep_delayed_stiff_fields_accurate(record_property):
    """Test that a delayed stiff field with the pattern agrees with the dense path."""
    pattern, state = _hutchinson_run('auto')
    dense, dense_state = _hutchinson_run('dense')
    record_property('dense_over_pattern', dense / pattern)
    assert np.linalg.norm(state - dense_state) / np.linalg.norm(dense_state) < 1e-5
