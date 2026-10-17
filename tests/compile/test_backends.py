"""Test the NumPy and Numba backends, compiled callables, kernel caching, and compiled stepper loops."""

import subprocess  # nosec B404: the cache tests start a second Python process
import sys
import textwrap

import numba
import numpy as np
import pytest

from skcomplex.compile import NumbaBackend, NumpyBackend, compile_system
from skcomplex.core import BackendError, Domain, SolverError, parameter, role
from skcomplex.model import Call, EuclideanSpace, Interaction, Jump, Noise, Rate, System, Update
from skcomplex.models.ecology import Population
from skcomplex.models.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.physics import HarmonicPotential, PairPotential, Particle, ReflectingBox
from skcomplex.runtime import Simulation
from skcomplex.solvers import RK4, EulerMaruyama, Gillespie, Iteration, VelocityVerlet

CUTOFF = 2.5


class LennardJones(PairPotential):
    """Lennard-Jones pair potential."""

    epsilon: float = parameter(1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(1.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


@numba.njit
def _logistic(t, n, r):
    """Return the logistic growth rate of a member."""
    return r * n * (1 - n / 100)


class CompiledGrowth(Interaction):
    """Logistic growth computed by a Numba-compiled function."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s):
        """Return the compiled rate."""
        return [Rate(s.target.n, Call(_logistic, s.target.n, s.r))]


class PythonGrowth(Interaction):
    """Logistic growth computed by a plain Python callable of the values."""

    target: str = role()

    def laws(self, s):
        """Return the Python rate."""
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: values[name] * (1 - values[name] / 100))]


class Halving(Interaction):
    """Discrete halving of every member."""

    target: str = role()

    def laws(self, s):
        """Return the next value `n / 2`."""
        return [Update(s.target.n, s.target.n / 2)]


class Volatility(Interaction):
    """Noise proportional to every member."""

    target: str = role()

    def laws(self, s):
        """Return the noise amplitude."""
        return [Rate(s.target.n, 0.1 * s.target.n), Noise(s.target.n, 0.2 * s.target.n)]


class Decay(Interaction):
    """Jump with a negative propensity, which is invalid."""

    target: str = role()

    def laws(self, s):
        """Return a jump with a negative rate."""
        return [Jump(-s.target.n, {s.target.n: -1})]


def _run_script(script):
    """Return the standard output of a Python script run in a new process."""
    command = [sys.executable, '-c', script]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603  # nosec B603


@pytest.fixture(scope='module')
def numba_backend(tmp_path_factory):
    """Return a Numba backend with a test cache directory."""
    return NumbaBackend(cache_dir=tmp_path_factory.mktemp('kernels'))


def _lennard_jones_gas(n=400, period=None):
    """Return a jittered lattice of Lennard-Jones particles in a reflecting box."""
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2 + 0.5
    size = side * 1.2 + 1.0
    return System(
        elements=[Particle.group('gas', n, x=grid + rng.normal(0, 0.05, (n, 2)), v=rng.normal(0, 1, (n, 2)))],
        interactions=[
            LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF, period=period),
            ReflectingBox('box', target='gas', size=size),
        ],
        space=EuclideanSpace(2),
    )


def _herds(interaction, n=5):
    """Return a group of populations with one interaction."""
    return System(elements=[Population.group('herds', n, n=np.linspace(1.0, 50.0, n))], interactions=[interaction])


def _epidemic():
    """Return a stochastic epidemic over a group of towns."""
    return System(
        elements=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


def test_compile_unknown_backend_raises_error():
    """Test that an unknown backend names the registered ones."""
    with pytest.raises(BackendError, match='auto, numpy, numba'):
        compile_system(System(elements=[Population('a')]), backend='gpu')


def test_compile_auto_backend_depends_on_group_size():
    """Test that `auto` picks Numba for 1000 members and NumPy for two single elements."""
    large = System(elements=[Particle.group('gas', 1000)], space=EuclideanSpace(2))
    small = System(elements=[Population('a'), Population('b')])
    assert compile_system(large).backend == 'numba'
    assert compile_system(small).backend == 'numpy'


def test_simulation_uses_given_backend(numba_backend):
    """Test that a simulation compiles with the backend it is given."""
    simulation = Simulation(_herds(Halving('half', target='herds')), backend=numba_backend).run(until=2)
    assert simulation.problem_.backend == 'numba'
    assert np.allclose(simulation.trajectory_['herds.n'][-1], np.linspace(1.0, 50.0, 5) / 4)


@pytest.mark.parametrize('period', [None, 24.0])
def test_backends_agree_on_gas_with_pair_interactions(numba_backend, period):
    """Test that drift and boundary agree between backends to 1e-12 relative error."""
    system = _lennard_jones_gas(period=period)
    reference, compiled = compile_system(system, backend=NumpyBackend()), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    for name in ('drift', 'boundary'):
        expected, result = getattr(reference, name)(0.0, x, p), getattr(compiled, name)(0.0, x, p)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_backends_agree_on_two_groups(numba_backend):
    """Test that pair forces between two groups agree between backends."""
    rng = np.random.default_rng(1)
    system = System(
        elements=[
            Particle.group('a', 150, x=rng.uniform(0, 12, (150, 2))),
            Particle.group('b', 100, x=rng.uniform(3, 15, (100, 2))),
        ],
        interactions=[LennardJones('lj', first='a', second='b', cutoff=CUTOFF)],
        space=EuclideanSpace(2),
    )
    reference, compiled = compile_system(system, backend='numpy'), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.drift(0.0, x, p), reference.drift(0.0, x, p), rtol=1e-12, atol=1e-12)


def test_backends_agree_on_discrete_update(numba_backend):
    """Test that the next state of a discrete map agrees between backends."""
    system = _herds(Halving('half', target='herds'))
    reference, compiled = compile_system(system, backend='numpy'), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.update(0.0, x, p), reference.update(0.0, x, p), rtol=1e-12)


def test_numba_pair_forces_conserve_momentum(numba_backend):
    """Test that compiled pair forces keep the total momentum of an isolated group."""
    system = System(
        elements=[_lennard_jones_gas().elements[0]],
        interactions=[LennardJones('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=EuclideanSpace(2),
    )
    problem = compile_system(system, backend=numba_backend)
    accelerations = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'gas.v')
    assert np.allclose(accelerations.sum(axis=0), 0.0, atol=1e-9)


def test_compiled_call_runs_inside_kernels(numba_backend):
    """Test that a Numba-compiled law value runs in compiled kernels with the NumPy results."""
    system = _herds(CompiledGrowth('growth', target='herds', r=0.5))
    reference, compiled = compile_system(system, backend='numpy'), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.drift(0.0, x, p), reference.drift(0.0, x, p), rtol=1e-12)
    assert compiled.accelerated is not None


def test_python_callable_runs_on_python_path(numba_backend):
    """Test that a plain Python callable compiles with the Numba backend through the Python path."""
    system = _herds(PythonGrowth('growth', target='herds'))
    compiled = compile_system(system, backend=numba_backend)
    expected = np.linspace(1.0, 50.0, 5) * (1 - np.linspace(1.0, 50.0, 5) / 100)
    assert np.allclose(compiled.drift(0.0, compiled.initial_state, compiled.initial_parameters), expected)
    assert compiled.accelerated is None


@pytest.mark.benchmark
def test_numba_kernels_are_cached_on_disk(tmp_path):
    """Test that compiling the same system in a second process is at least five times faster."""
    script = textwrap.dedent(
        f"""
        import time
        from skcomplex.compile import NumbaBackend, compile_system
        from skcomplex.models.physics import ideal_gas
        start = time.perf_counter()
        problem = compile_system(ideal_gas(2000, random_state=0), backend=NumbaBackend(cache_dir={str(tmp_path)!r}))
        problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
        problem.accelerated.rk4(0.0, problem.initial_state, problem.initial_parameters, 0.0, 1.0)
        print(time.perf_counter() - start)
        """,
    )
    durations = [float(_run_script(script)) for _ in range(2)]
    assert durations[1] * 5 <= durations[0]


def test_second_process_loads_kernels_from_cache(tmp_path):
    """Test that a second process loads every generated kernel from the disk cache without compiling it."""
    script = textwrap.dedent(
        f"""
        import sys
        from skcomplex.compile import NumbaBackend, compile_system
        from skcomplex.models.physics import ideal_gas
        problem = compile_system(ideal_gas(2000, random_state=0), backend=NumbaBackend(cache_dir={str(tmp_path)!r}))
        problem.drift(0.0, problem.initial_state, problem.initial_parameters)
        problem.boundary(0.0, problem.initial_state, problem.initial_parameters)
        module = next(module for name, module in sys.modules.items() if name.startswith('skcomplex_kernels_'))
        dispatchers = [value for value in vars(module).values() if hasattr(value, 'stats')]
        hits = sum(sum(dispatcher.stats.cache_hits.values()) for dispatcher in dispatchers)
        print(hits, sum(sum(dispatcher.stats.cache_misses.values()) for dispatcher in dispatchers))
        """,
    )
    runs = [_run_script(script).split() for _ in range(2)]
    _, first_misses = map(int, runs[0])
    second_hits, second_misses = map(int, runs[1])
    assert first_misses > 0
    assert second_misses == 0
    assert second_hits > 0


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_rk4_matches_exponential(backend, numba_backend):
    """Test that RK4 with step 0.001 integrates exponential growth to 1e-10 relative error."""

    class Growth(Interaction):
        """Exponential growth of every member."""

        target: str = role()

        def laws(self, s):
            """Return the rate `n`."""
            return [Rate(s.target.n, s.target.n)]

    problem = compile_system(
        System(elements=[Population.group('a', 2, n=1.0)], interactions=[Growth('g', target='a')]),
        backend=numba_backend if backend == 'numba' else backend,
    )
    stepper = RK4(dt=0.001)
    stepper.initialize(problem, np.random.default_rng(0))
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
    assert np.allclose(state, np.e, rtol=1e-10)


def test_compiled_gillespie_is_reproducible(numba_backend):
    """Test that the compiled Gillespie loop gives equal trajectories for equal random states."""
    runs = [
        Simulation(_epidemic(), solver=Gillespie(), backend=numba_backend, dt=1.0, random_state=0).run(until=30)
        for _ in range(2)
    ]
    assert runs[0].problem_.accelerated is not None
    assert np.array_equal(runs[0].trajectory_.states, runs[1].trajectory_.states)
    assert np.allclose(runs[0].trajectory_.states.sum(axis=1), 1000)


def test_compiled_gillespie_negative_propensity_raises_error(numba_backend):
    """Test that a negative propensity in the compiled loop raises a solver error."""
    with pytest.raises(SolverError, match='non-negative'):
        Simulation(_herds(Decay('decay', target='herds')), backend=numba_backend, dt=1.0).run(until=1)


def test_compiled_euler_maruyama_is_reproducible(numba_backend):
    """Test that the compiled Euler-Maruyama loop is reproducible and keeps prices positive."""
    runs = [
        Simulation(
            _herds(Volatility('vol', target='herds')),
            solver=EulerMaruyama(dt=1e-3),
            backend=numba_backend,
            dt=0.1,
            random_state=5,
        ).run(until=2)
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].trajectory_.states, runs[1].trajectory_.states)
    assert np.all(runs[0].trajectory_.states > 0)


def test_compiled_iteration_matches_numpy(numba_backend):
    """Test that the compiled discrete loop matches the NumPy loop."""
    results = [
        Simulation(_herds(Halving('half', target='herds')), solver=Iteration(), backend=backend, dt=3)
        .run(until=9)
        .trajectory_.states
        for backend in ('numpy', numba_backend)
    ]
    assert np.allclose(results[0], results[1])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_velocity_verlet_conserves_energy(backend, numba_backend):
    """Test that velocity Verlet keeps the energy of an oscillator within 1e-3 over 1000 time units."""
    system = System(
        elements=[Particle.group('p', 1, x=[[1.0]], v=[[0.0]])],
        interactions=[HarmonicPotential('trap', target='p')],
        space=EuclideanSpace(1),
    )
    simulation = Simulation(
        system,
        solver=VelocityVerlet(dt=0.01),
        backend=numba_backend if backend == 'numba' else backend,
        dt=10.0,
    )
    trajectory = simulation.run(until=1000).trajectory_
    energy = 0.5 * trajectory['p.v'][:, 0, 0] ** 2 + 0.5 * trajectory['p.x'][:, 0, 0] ** 2
    assert np.allclose(energy, energy[0], rtol=1e-3)


def test_velocity_verlet_needs_position_pairs():
    """Test that velocity Verlet names the states that are neither positions nor velocities."""
    with pytest.raises(SolverError, match=r'herds\.n are neither'):
        Simulation(_herds(CompiledGrowth('growth', target='herds')), solver=VelocityVerlet()).run(until=1)
