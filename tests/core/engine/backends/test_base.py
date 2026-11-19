"""Test the NumPy and Numba backends, compiled callables, kernel caching, and compiled stepper loops."""

import inspect
from collections import defaultdict
from types import SimpleNamespace
from typing import Annotated

import numba
import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from skcomplex.core.engine import backends as package
from skcomplex.core.engine._utils import vector
from skcomplex.core.engine.backends import (
    BaseBackend,
    JaxBackend,
    LoweredFunctionsSpec,
    NumbaBackend,
    NumpyBackend,
    _numba,
)
from skcomplex.core.engine.problems._lowered_kernels_spec import LoweredKernelsSpec
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    IterationSolver,
    RK4Solver,
    StepSpec,
    VelocityVerletSolver,
)
from skcomplex.core.enums import FormKind, OrderKind
from skcomplex.core.errors import BackendWarning, SolverError
from skcomplex.core.specs import BaseSpec, Role
from skcomplex.core.systems import BaseSystem, System
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.systems.physics.statistical_mechanics import ContactProcessSystem
from skcomplex.systems.physics.waves import HarmonicPotentialInteraction

from ....conftest import BodyElement

CUTOFF = 2.5


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x


def _still(label):
    """Return a zero rate of the value `x` of an element, which keeps it a state."""
    return Interaction(f'{label}_still', roles={'target': label}, laws=lambda s: [RateLaw(s.target.x, 0)])


def _inertia(label):
    """Return a zero acceleration of the positions `x` of a group, which move at their velocities `x'`."""
    return Interaction(
        f'{label}_inertia',
        roles={'target': label},
        laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0)],
    )


def _problem(system, backend='auto'):
    """Return the problem a system compiles to on a backend."""
    prepared = system.set_params(backend=backend).prepare()
    return prepared.system_._problem if isinstance(prepared, BaseSystem) else prepared._problem


def _run(system, until, solver='auto', backend='auto', **settings):
    """Return the trajectory of a system run with a solver on a backend."""
    return system.set_params(solver=solver, backend=backend).prepare().simulate(until=until, **settings)


class LennardJonesInteraction(Interaction):
    """Lennard-Jones pair potential between the members of two roles within the cutoff."""

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
        target: Role,
        r: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the compiled rate."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.target.x, CallSpec(_logistic, (s.target.x, s.r)))]


class PythonGrowthInteraction(Interaction):
    """Logistic growth computed by a plain Python callable of the values."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the Python rate."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            name = f'{self.target}.x'
            self.laws_ = [RateLaw(s.target.x, lambda t, values: values[name] * (1 - values[name] / 100))]


class HalvingInteraction(Interaction):
    """Discrete halving of every member."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next value `n / 2`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [UpdateLaw(s.target.x, s.target.x / 2)]


class VolatilityInteraction(Interaction):
    """Noise proportional to every member."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise amplitude."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.target.x, 0.1 * s.target.x), NoiseLaw(s.target.x, 0.2 * s.target.x)]


class DecayInteraction(Interaction):
    """Jump with a negative propensity, which is invalid."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a jump with a negative rate."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [JumpLaw(-s.target.x, {s.target.x: -1})]


@pytest.fixture(scope='module')
def numba_backend(tmp_path_factory):
    """Return a Numba backend with a test cache directory."""
    return NumbaBackend(cache_dir=tmp_path_factory.mktemp('kernels'))


def _lennard_jones_gas(n=400, boundary='reflecting'):
    """Return a jittered lattice of Lennard-Jones particles in a box of a boundary."""
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n] * 1.2 + 0.5
    size = side * 1.2 + 1.0
    gas = BodyElement('gas', x=grid + rng.normal(0, 0.05, (n, 2)), values={"x'": rng.normal(0, 1, (n, 2))}, size=n)
    return System(
        elements=[gas],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(extent=(size, size), boundary=boundary),
    )


def _herds(interaction, n=5):
    """Return a group of populations with one interaction."""
    return System(
        elements=[Element('herds', values={'x': np.linspace(1.0, 50.0, n)}, size=n)],
        interactions=[interaction],
    )


def _epidemic():
    """Return a stochastic epidemic over a group of towns."""
    return System(
        elements=[AmountElement('S', x=990), AmountElement('I', x=10), AmountElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_backend_runs_the_kernels_of_a_form(backend):
    """Test that a checked and prepared backend returns the functions of the form of the kernels it runs."""
    kernels = LoweredKernelsSpec(defaultdict(list), {}, {}, form=FormKind.ODE, size=2)
    lowering = backend().prepare().run(kernels).functions_
    assert isinstance(lowering, LoweredFunctionsSpec)
    assert np.array_equal(lowering.drift(0.0, np.ones(2), np.zeros(0)), np.zeros(2))


def test_compile_auto_backend_depends_on_group_size():
    """Test that `auto` picks Numba for 1000 members and NumPy for two single elements."""
    large = System(
        elements=[BodyElement('gas', x=np.zeros((1000, 2)), size=1000)],
        interactions=[_inertia('gas')],
        space=ContinuousSpace(n_dim=2),
    )
    small = System(elements=[PopulationElement('a'), PopulationElement('b')], interactions=[_still('a'), _still('b')])
    assert _problem(large).backend == 'numba'
    assert _problem(small).backend == 'numpy'


def test_simulation_uses_given_backend(numba_backend):
    """Test that a simulation compiles with the backend it is given."""
    trajectory = _run(_herds(HalvingInteraction('half', target='herds')), 2, backend=numba_backend)
    assert trajectory.problem.backend == 'numba'
    assert np.allclose(trajectory.get_values('herds.x')[-1], np.linspace(1.0, 50.0, 5) / 4)


@pytest.mark.parametrize('boundary', ['reflecting', 'periodic'])
def test_backends_agree_on_gas_with_pair_interactions(numba_backend, boundary):
    """Test that drift and boundary agree between backends to 1e-12 relative error."""
    system = _lennard_jones_gas(boundary=boundary)
    reference, compiled = (
        _problem(system, NumpyBackend()),
        _problem(system, numba_backend),
    )
    x, p = reference.initial_state, reference.initial_parameters
    for name in ('drift', 'boundary'):
        expected, result = getattr(reference, name)(0.0, x, p), getattr(compiled, name)(0.0, x, p)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_backends_agree_on_two_groups(numba_backend):
    """Test that pair forces between two groups agree between backends."""
    rng = np.random.default_rng(1)
    system = System(
        elements=[
            BodyElement('a', x=rng.uniform(0, 12, (150, 2)), values={"x'": np.zeros((150, 2))}, size=150),
            BodyElement('b', x=rng.uniform(3, 15, (100, 2)), values={"x'": np.zeros((100, 2))}, size=100),
        ],
        interactions=[LennardJonesInteraction('lj', first='a', second='b', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    reference, compiled = (
        _problem(system, NumpyBackend()),
        _problem(system, numba_backend),
    )
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.drift(0.0, x, p), reference.drift(0.0, x, p), rtol=1e-12, atol=1e-12)


def test_backends_agree_on_discrete_update(numba_backend):
    """Test that the next state of a discrete map agrees between backends."""
    system = _herds(HalvingInteraction('half', target='herds'))
    reference, compiled = (
        _problem(system, NumpyBackend()),
        _problem(system, numba_backend),
    )
    x, p = reference.initial_state, reference.initial_parameters
    assert np.allclose(compiled.update(0.0, x, p), reference.update(0.0, x, p), rtol=1e-12)


def test_numba_pair_forces_conserve_momentum(numba_backend):
    """Test that compiled pair forces keep the total momentum of an isolated group."""
    gas = _lennard_jones_gas().elements[0]
    system = System(
        elements=[gas],
        interactions=[LennardJonesInteraction('lj', first='gas', second='gas', cutoff=CUTOFF)],
        space=ContinuousSpace(n_dim=2),
    )
    problem = _problem(system, numba_backend)
    accelerations = problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        "gas.x'",
    )
    assert np.allclose(accelerations.sum(axis=0), 0.0, atol=1e-9)


def test_compiled_call_runs_inside_kernels(numba_backend):
    """Test that a Numba-compiled law value runs in compiled kernels with the NumPy results."""
    system = _herds(CompiledGrowthInteraction('growth', target='herds', r=0.5))
    reference, compiled = (
        _problem(system, NumpyBackend()),
        _problem(system, numba_backend),
    )
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

        def __init__(self, label: str, *, x: object = 1.0, y: object = 0.0) -> None:
            super().__init__(label)
            self.x = x
            self.y = y

    def laws(s):
        """Return the rotation of the pair by a call."""
        x, y = s.target.x, s.target.y
        return [UpdateLaw(vector([x, y]), CallSpec(rotate, (x, y)))]

    system = System(
        elements=[PairElement('pair')],
        interactions=[Interaction('turn', roles={'target': 'pair'}, laws=laws)],
    )
    trajectory = _run(system, 4, backend=NumpyBackend())
    assert trajectory.get_values('pair.x').tolist() == [1.0, 0.0, -1.0, 0.0, 1.0]
    assert [calls.count(float(t)) for t in (1, 2, 3)] == [1, 1, 1]


def test_python_callable_runs_on_python_path(numba_backend):
    """Test that a plain Python callable compiles with the Numba backend through the Python path."""
    system = _herds(PythonGrowthInteraction('growth', target='herds'))
    compiled = _problem(system, numba_backend)
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
            target: Role,
        ) -> None:
            super().__init__(label)
            self.target = target

        def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
            """Set the rate `x`."""
            super()._prepare(context)
            if isinstance(context, SimpleNamespace):
                s = context
                self.laws_ = [RateLaw(s.target.x, s.target.x)]

    system = System(
        elements=[Element('a', values={'x': np.ones(2)}, size=2)],
        interactions=[GrowthInteraction('g', target='a')],
    )
    problem = _problem(system, numba_backend if backend == 'numba' else NumpyBackend())
    stepper = RK4Solver(dt=0.001)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_
    assert np.allclose(state, np.e, rtol=1e-10)


def test_compiled_gillespie_is_reproducible(numba_backend):
    """Test that the compiled Gillespie loop gives equal trajectories for equal random states."""
    runs = [_run(_epidemic(), 30, GillespieSolver(), numba_backend, dt=1.0, random_state=0) for _ in range(2)]
    assert runs[0].problem.accelerated is not None
    assert np.array_equal(runs[0].states, runs[1].states)
    assert np.allclose(runs[0].states.sum(axis=1), 1000)


def test_compiled_gillespie_negative_propensity_raises_error(numba_backend):
    """Test that a negative propensity in the compiled loop raises a solver error."""
    with pytest.raises(SolverError, match='non-negative'):
        _run(_herds(DecayInteraction('decay', target='herds')), 1, backend=numba_backend, dt=1.0)


def test_compiled_euler_maruyama_is_reproducible(numba_backend):
    """Test that the compiled Euler-Maruyama loop is reproducible and keeps prices positive."""
    runs = [
        _run(
            _herds(VolatilityInteraction('vol', target='herds')),
            2,
            EulerMaruyamaSolver(dt=1e-3),
            numba_backend,
            dt=0.1,
            random_state=5,
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].states, runs[1].states)
    assert np.all(runs[0].states > 0)


def test_compiled_iteration_matches_numpy(numba_backend):
    """Test that the compiled discrete loop matches the NumPy loop."""
    results = [
        _run(_herds(HalvingInteraction('half', target='herds')), 9, IterationSolver(), backend, dt=3).states
        for backend in (NumpyBackend(), numba_backend)
    ]
    assert np.allclose(results[0], results[1])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_velocity_verlet_conserves_energy(backend, numba_backend):
    """Test that velocity Verlet keeps the energy of an oscillator within 1e-3 over 1000 time units."""
    system = System(
        elements=[BodyElement('p', x=[[1.0]], values={"x'": [[0.0]]}, size=1)],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    runner = numba_backend if backend == 'numba' else NumpyBackend()
    trajectory = _run(system, 1000, VelocityVerletSolver(dt=0.01), runner, dt=10.0)
    energy = 0.5 * trajectory.get_values("p.x'")[:, 0, 0] ** 2 + 0.5 * trajectory.get_values('p.x')[:, 0, 0] ** 2
    assert np.allclose(energy, energy[0], rtol=1e-3)


def test_velocity_verlet_needs_position_pairs():
    """Test that velocity Verlet names the states that are neither positions nor velocities."""
    with pytest.raises(SolverError, match=r'herds\.x are neither'):
        _run(_herds(CompiledGrowthInteraction('growth', target='herds')), 1, VelocityVerletSolver())


def test_backend_substituted_for_a_form_is_reported():
    """Test that a lattice system compiled with Numba warns that its form runs on NumPy, and records NumPy for it."""
    with pytest.warns(BackendWarning, match='lattice dynamics form runs on the numpy backend'):
        problem = _problem(ContactProcessSystem(), NumbaBackend())
    assert problem.backends['LATTICE'] == 'numpy'


def test_kernel_falling_back_to_python_is_reported(numba_backend):
    """Test that a kernel the Numba backend cannot compile warns once, naming the law kind, and records Python."""
    system = _herds(PythonGrowthInteraction('growth', target='herds'))
    with pytest.warns(BackendWarning, match='runs the RateLaw laws in Python') as found:
        problem = _problem(system, numba_backend)
    assert len([item for item in found if item.category is BackendWarning]) == 1
    assert problem.backends == {'ODE': 'numba', 'RateLaw': 'python'}


def test_backends_derive_from_the_backend_base():
    """Test that the code backends of the engine derive from the backend base of the core."""
    assert issubclass(NumbaBackend, BaseBackend)
    assert issubclass(NumpyBackend, BaseBackend)


def test_backends_support_the_tags_of_the_classes_they_support():
    """Test that a backend supports the tags of the classes it supports."""
    assert JaxBackend.get_supported_tags().space == {'none'}
    assert 'noise' not in JaxBackend.get_supported_tags().laws
    assert 'noise' in NumpyBackend.get_supported_tags().laws


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_backends_support_every_order(backend):
    """Test that every backend supports integer and fractional orders."""
    assert backend._get_support().orders == set(OrderKind)


def test_public_names_are_classes_under_the_bases():
    """Every public name is a backend or a value a backend returns."""
    for name in package.__all__:
        item = getattr(package, name)
        assert inspect.isclass(item), name
        assert issubclass(item, BaseBackend | BaseSpec), name


def test_backends_name_themselves_and_their_support():
    """Every concrete backend answers the hooks of its name and of what it supports."""
    for backend in (package.NumpyBackend, package.NumbaBackend, package.JaxBackend):
        assert isinstance(backend._get_name(), str)
        assert backend._get_support().laws
