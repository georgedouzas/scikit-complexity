"""Test the NumPy and Numba backends, compiled callables, kernel caching, and compiled stepper loops."""

from collections import defaultdict
from typing import Annotated

import numba
import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import CompartmentElement, Element, GroupElement
from skcomplex.core.interactions import Interaction, PairCouplingInteraction
from skcomplex.core.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine._utils import vector
from skcomplex.engine.backends import BaseBackend, NumbaBackend, NumpyBackend, _numba
from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import LoweredKernelsSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    IterationSolver,
    RK4Solver,
    VelocityVerletSolver,
)
from skcomplex.errors import BackendWarning, SolverError
from skcomplex.models import create
from skcomplex.models.biology.common.populations import PopulationElement
from skcomplex.models.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.models.physics.common.particles import HarmonicPotentialInteraction, MassPointElement
from skcomplex.specs import CallSpec, LoweredFunctionsSpec, ParameterSpec, RoleSpec, StateSpec, StepSpec, VisualSpec
from skcomplex.types import FormKind

CUTOFF = 2.5


class LennardJonesInteraction(PairCouplingInteraction):
    """Lennard-Jones pair potential."""

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            potential=self._get_pair_energy,
            period=period,
        )
        self.epsilon = epsilon
        self.sigma = sigma

    def _get_pair_energy(self, s, r):
        """Return `4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)`."""
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


@numba.njit
def _logistic(t, n, r):
    """Return the logistic growth rate of a member."""
    return r * n * (1 - n / 100)


class CompiledGrowthInteraction(Interaction):
    """Logistic growth computed by a Numba-compiled function."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        """Return the compiled rate."""
        return [RateLaw(s.target.n, CallSpec(_logistic, (s.target.n, s.r)))]


class PythonGrowthInteraction(Interaction):
    """Logistic growth computed by a plain Python callable of the values."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the Python rate."""
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: values[name] * (1 - values[name] / 100))]


class HalvingInteraction(Interaction):
    """Discrete halving of every member."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the next value `n / 2`."""
        return [UpdateLaw(s.target.n, s.target.n / 2)]


class VolatilityInteraction(Interaction):
    """Noise proportional to every member."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the noise amplitude."""
        return [RateLaw(s.target.n, 0.1 * s.target.n), NoiseLaw(s.target.n, 0.2 * s.target.n)]


class DecayInteraction(Interaction):
    """Jump with a negative propensity, which is invalid."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return a jump with a negative rate."""
        return [JumpLaw(-s.target.n, {s.target.n: -1})]


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
        elements=[
            GroupElement(
                'gas',
                element=MassPointElement,
                size=n,
                values={'x': grid + rng.normal(0, 0.05, (n, 2)), 'x_d1': rng.normal(0, 1, (n, 2))},
            ),
        ],
        interactions=[
            LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF, period=period),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='reflecting'),
    )


def _herds(interaction, n=5):
    """Return a group of populations with one interaction."""
    return System(
        elements=[GroupElement('herds', element=PopulationElement, size=n, values={'n': np.linspace(1.0, 50.0, n)})],
        interactions=[interaction],
    )


def _epidemic():
    """Return a stochastic epidemic over a group of towns."""
    return System(
        elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_backend_runs_the_kernels_of_a_form(backend):
    """Test that a checked and prepared backend returns the functions of the form of the kernels it runs."""
    kernels = LoweredKernelsSpec(defaultdict(list), {}, {}, form=FormKind.ODE, size=2)
    lowering = backend().check().prepare().run(kernels).functions_
    assert isinstance(lowering, LoweredFunctionsSpec)
    assert np.array_equal(lowering.drift(0.0, np.ones(2), np.zeros(0)), np.zeros(2))


def test_compile_auto_backend_depends_on_group_size():
    """Test that `auto` picks Numba for 1000 members and NumPy for two single elements."""
    large = System(elements=[GroupElement('gas', element=MassPointElement, size=1000)], space=ContinuousSpace(n_dim=2))
    small = System(elements=[PopulationElement('a'), PopulationElement('b')])
    assert compile_system(large).backend == 'numba'
    assert compile_system(small).backend == 'numpy'


def test_simulation_uses_given_backend(numba_backend):
    """Test that a simulation compiles with the backend it is given."""
    simulation = (
        Simulation(_herds(HalvingInteraction('half', target='herds')), backend=numba_backend)
        .check()
        .prepare()
        .run(until=2)
    )
    assert simulation.problem_.backend == 'numba'
    assert np.allclose(simulation.trajectory_.values('herds.n')[-1], np.linspace(1.0, 50.0, 5) / 4)


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
            GroupElement('a', element=MassPointElement, size=150, values={'x': rng.uniform(0, 12, (150, 2))}),
            GroupElement('b', element=MassPointElement, size=100, values={'x': rng.uniform(3, 15, (100, 2))}),
        ],
        interactions=[LennardJonesInteraction('lj', first='a', second='b', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    reference, compiled = compile_system(system, backend=NumpyBackend()), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.drift(0.0, x, p), reference.drift(0.0, x, p), rtol=1e-12, atol=1e-12)


def test_backends_agree_on_discrete_update(numba_backend):
    """Test that the next state of a discrete map agrees between backends."""
    system = _herds(HalvingInteraction('half', target='herds'))
    reference, compiled = compile_system(system, backend=NumpyBackend()), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.update(0.0, x, p), reference.update(0.0, x, p), rtol=1e-12)


def test_numba_pair_forces_conserve_momentum(numba_backend):
    """Test that compiled pair forces keep the total momentum of an isolated group."""
    system = System(
        elements=[_lennard_jones_gas().elements[0]],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = compile_system(system, backend=numba_backend)
    accelerations = problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        'gas.x_d1',
    )
    assert np.allclose(accelerations.sum(axis=0), 0.0, atol=1e-9)


def test_compiled_call_runs_inside_kernels(numba_backend):
    """Test that a Numba-compiled law value runs in compiled kernels with the NumPy results."""
    system = _herds(CompiledGrowthInteraction('growth', target='herds', r=0.5))
    reference, compiled = compile_system(system, backend=NumpyBackend()), compile_system(system, backend=numba_backend)
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.drift(0.0, x, p), reference.drift(0.0, x, p), rtol=1e-12)
    assert compiled.accelerated is not None


def test_numpy_backend_calls_vector_call_once_per_step():
    """Test that a call with a vector target runs once per step, not once per component."""
    calls = []

    def rotate(t, x, y):
        calls.append(t)
        return [-y, x]

    class PairElement(Element):
        """Pair element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the pair element."""
            return [UpdateLaw(vector([s.x, s.y]), CallSpec(rotate, (s.x, s.y)))]

    trajectory = (
        Simulation(System(elements=[PairElement('pair')]), backend=NumpyBackend())
        .check()
        .prepare()
        .run(until=4)
        .trajectory_
    )
    assert trajectory.values('pair.x').tolist() == [1.0, 0.0, -1.0, 0.0, 1.0]
    assert [calls.count(float(t)) for t in (1, 2, 3)] == [1, 1, 1]


def test_python_callable_runs_on_python_path(numba_backend):
    """Test that a plain Python callable compiles with the Numba backend through the Python path."""
    system = _herds(PythonGrowthInteraction('growth', target='herds'))
    compiled = compile_system(system, backend=numba_backend)
    expected = np.linspace(1.0, 50.0, 5) * (1 - np.linspace(1.0, 50.0, 5) / 100)
    assert np.allclose(compiled.drift(0.0, compiled.initial_state, compiled.initial_parameters), expected)
    assert compiled.accelerated is None


def test_cached_kernels_are_keyed_on_the_compiled_helpers(tmp_path, monkeypatch):
    """Test that the same generated source is cached apart when the compiled helpers it imports change."""
    text = 'VALUE = 1\n'
    NumbaBackend._load(text, {}, tmp_path)
    monkeypatch.setattr(_numba, 'HELPERS_FINGERPRINT', 'other helpers')
    NumbaBackend._load(text, {}, tmp_path)
    assert len(list(tmp_path.glob('skcomplex_kernels_*.py'))) == 2


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_rk4_matches_exponential(backend, numba_backend):
    """Test that RK4 with step 0.001 integrates exponential growth to 1e-10 relative error."""

    class GrowthInteraction(Interaction):
        """Exponential growth of every member."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[str, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return the rate `n`."""
            return [RateLaw(s.target.n, s.target.n)]

    problem = compile_system(
        System(
            elements=[GroupElement('a', element=PopulationElement, size=2, values={'n': 1.0})],
            interactions=[GrowthInteraction('g', target='a')],
        ),
        backend=numba_backend if backend == 'numba' else NumpyBackend(),
    )
    stepper = RK4Solver(dt=0.001)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_
    assert np.allclose(state, np.e, rtol=1e-10)


def test_compiled_gillespie_is_reproducible(numba_backend):
    """Test that the compiled Gillespie loop gives equal trajectories for equal random states."""
    runs = [
        (
            Simulation(_epidemic(), solver=GillespieSolver(), backend=numba_backend, dt=1.0, random_state=0)
            .check()
            .prepare()
            .run(until=30)
        )
        for _ in range(2)
    ]
    assert runs[0].problem_.accelerated is not None
    assert np.array_equal(runs[0].trajectory_.states, runs[1].trajectory_.states)
    assert np.allclose(runs[0].trajectory_.states.sum(axis=1), 1000)


def test_compiled_gillespie_negative_propensity_raises_error(numba_backend):
    """Test that a negative propensity in the compiled loop raises a solver error."""
    with pytest.raises(SolverError, match='non-negative'):
        (
            Simulation(_herds(DecayInteraction('decay', target='herds')), backend=numba_backend, dt=1.0)
            .check()
            .prepare()
            .run(until=1)
        )


def test_compiled_euler_maruyama_is_reproducible(numba_backend):
    """Test that the compiled Euler-Maruyama loop is reproducible and keeps prices positive."""
    runs = [
        Simulation(
            _herds(VolatilityInteraction('vol', target='herds')),
            solver=EulerMaruyamaSolver(dt=1e-3),
            backend=numba_backend,
            dt=0.1,
            random_state=5,
        )
        .check()
        .prepare()
        .run(until=2)
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].trajectory_.states, runs[1].trajectory_.states)
    assert np.all(runs[0].trajectory_.states > 0)


def test_compiled_iteration_matches_numpy(numba_backend):
    """Test that the compiled discrete loop matches the NumPy loop."""
    results = [
        (
            Simulation(
                _herds(HalvingInteraction('half', target='herds')),
                solver=IterationSolver(),
                backend=backend,
                dt=3,
            )
            .check()
            .prepare()
        )
        .run(until=9)
        .trajectory_.states
        for backend in (NumpyBackend(), numba_backend)
    ]
    assert np.allclose(results[0], results[1])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_velocity_verlet_conserves_energy(backend, numba_backend):
    """Test that velocity Verlet keeps the energy of an oscillator within 1e-3 over 1000 time units."""
    system = System(
        elements=[GroupElement('p', element=MassPointElement, size=1, values={'x': [[1.0]], 'x_d1': [[0.0]]})],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    simulation = (
        Simulation(
            system,
            solver=VelocityVerletSolver(dt=0.01),
            backend=numba_backend if backend == 'numba' else NumpyBackend(),
            dt=10.0,
        )
        .check()
        .prepare()
    )
    trajectory = simulation.run(until=1000).trajectory_
    energy = 0.5 * trajectory.values('p.x_d1')[:, 0, 0] ** 2 + 0.5 * trajectory.values('p.x')[:, 0, 0] ** 2
    assert np.allclose(energy, energy[0], rtol=1e-3)


def test_velocity_verlet_needs_position_pairs():
    """Test that velocity Verlet names the states that are neither positions nor velocities."""
    with pytest.raises(SolverError, match=r'herds\.n are neither'):
        Simulation(
            _herds(CompiledGrowthInteraction('growth', target='herds')),
            solver=VelocityVerletSolver(),
        ).check().prepare().run(
            until=1,
        )


def test_backend_substituted_for_a_form_is_reported():
    """Test that a lattice model compiled with Numba warns that its form runs on NumPy, and records NumPy for it."""
    with pytest.warns(BackendWarning, match='lattice dynamics form runs on the numpy backend'):
        problem = compile_system(create('contact_process'), backend=NumbaBackend())
    assert problem.backends['LATTICE'] == 'numpy'


def test_kernel_falling_back_to_python_is_reported(numba_backend):
    """Test that a kernel the Numba backend cannot compile warns once, naming the law kind, and records Python."""
    system = _herds(PythonGrowthInteraction('growth', target='herds'))
    with pytest.warns(BackendWarning, match='runs the RateLaw laws in Python') as found:
        problem = compile_system(system, backend=numba_backend)
    assert len([item for item in found if item.category is BackendWarning]) == 1
    assert problem.backends == {'ODE': 'numba', 'RateLaw': 'python'}


def test_backends_derive_from_the_backend_base():
    """Test that the code backends of the engine derive from the backend base of the core."""
    assert issubclass(NumbaBackend, BaseBackend)
    assert issubclass(NumpyBackend, BaseBackend)
