"""Test the `ScipyODESolver` class."""

import dataclasses
import math
import time
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ceil, sin
from scipy.integrate import solve_ivp
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, EventLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian, mean, step, total, where
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.problems import RunStateSpec
from skcomplex.core.engine.solvers import CrankNicolsonSolver, RK4Solver, ScipyODESolver, StepSpec, _scipy_ode
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.biology.ecology import FisherKPPSystem
from skcomplex.systems.biology.physiology import FitzhughNagumoSystem
from skcomplex.systems.chemistry.kinetics import NanoparticleCoronaSystem
from skcomplex.systems.mathematics.dynamical_systems import FiniteTimeBlowUpSystem, TakensBogdanovFormSystem
from skcomplex.systems.physics.heat import HeatDiffusionSystem
from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem

STIFF_SIZE = 4000
SIZE = 400


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    return system.prepare()._problem


def _system(label, laws, size=None, space=None, interactions=(), **settings):
    """Return one element of values, a group of a size, or a field on a space, with laws of the namespace.

    The values are the settings that are not `solver` or `backend`.
    """
    runners = {name: settings.pop(name) for name in ('solver', 'backend') if name in settings}
    element = Element(label, values=settings, size=size, space=space)
    own = Interaction('laws', roles={'target': label}, laws=laws)
    return System(elements=[element], interactions=[own, *interactions], **runners)


def _fisher_kpp(s):
    """Return the diffusion and the logistic growth of the density, `du/dt = D laplacian(u) + r u (1 - u)`."""
    u = s.target.u
    return [RateLaw(u.diff(s.t), s.target.D * laplacian(u) + s.target.r * u * (1 - u))]


def _diffusive_hutchinson(s):
    """Return the diffusion and the delayed logistic growth of the density."""
    u = s.target.u
    return [RateLaw(u.diff(s.t), s.target.D * laplacian(u) + s.target.r * u * (1 - u(t=s.t - 1.0)))]


@pytest.mark.parametrize('method', ['RK45', 'DOP853', 'LSODA'])
def test_scipy_ode_matches_exponential(method):
    """Test that the ODE stepper integrates exponential growth accurately."""
    problem = _growth_problem()
    stepper = ScipyODESolver(method=method, rtol=1e-9, atol=1e-12)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_
    assert state[0] == pytest.approx(np.exp(2.0), rel=1e-6)


class ExplosionInteraction(Interaction):
    """Growth `dn/dt = n**2`, which blows up in finite time."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
    ) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `n**2`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_: list[BaseLaw] = [RateLaw(x.diff(context.t), x**2)]


def test_scipy_ode_failure_raises_error():
    """Test that an integrator failure raises a named error."""
    system = System(elements=[PopulationElement('a', x=1)], interactions=[ExplosionInteraction('boom', target='a')])
    problem = system.prepare()._problem
    stepper = ScipyODESolver()
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    with pytest.raises(SolverError, match='failed'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))


def test_simulation_runs_stiff_system_with_default_solver():
    """Test that a stiff system built directly runs fast with the default solver and matches its registered solver."""
    start = time.perf_counter()
    built = clone(NanoparticleCoronaSystem().prepare().system_).set_params(solver='auto')
    default = built.prepare().simulate(until=10800)
    assert time.perf_counter() - start < 5
    registered = NanoparticleCoronaSystem().prepare().simulate(until=10800, random_state=0)
    np.testing.assert_allclose(default.states[-1], registered.states[-1], rtol=1e-4, atol=1e-12)


def test_simulation_default_solver_keeps_accuracy_of_non_stiff_system():
    """Test that a harmonic oscillator run with the default solver matches the analytic solution within 1e-5."""
    trajectory = HarmonicOscillatorSystem().prepare().simulate(until=10)
    np.testing.assert_allclose(trajectory.states[-1], [np.cos(10), -np.sin(10)], rtol=1e-5)


def _relaxing_group(size):
    """Return a group whose members relax to their mean, so every member reads every other one."""

    def relax(s):
        """Return the relaxation of every member to the mean."""
        x = s.target.x
        return [RateLaw(x.diff(s.t), -(x - mean(x)))]

    return _system('cells', relax, size=size, x=np.linspace(0.0, 1.0, size))


def _stepped_solver(problem, until, dt):
    """Return the default ODE solver after it advanced a problem frame by frame until a time."""
    solver = ScipyODESolver(until=until).prepare(problem)
    state, frames = solver.initial_state_, round(until / dt)
    for frame in range(frames):
        step = StepSpec(frame * dt, state, problem.initial_parameters, (frame + 1) * dt)
        state = solver.run(step, final=frame + 1 == frames).state_
    return solver


def test_default_solver_leaves_lsoda_for_states_its_dense_matrix_cannot_hold():
    """Test that 10^4 coupled states leave LSODA, a banded field of 6400 states keeps it, and LSODA names the limit."""
    trajectory = _relaxing_group(10_000).prepare().simulate(until=1, dt=0.5)
    assert _stepped_solver(trajectory.problem, 1, 0.5).method_ != 'LSODA'
    start = np.linspace(0.0, 1.0, 10_000)
    np.testing.assert_allclose(trajectory.states[-1], 0.5 + (start - 0.5) * np.exp(-1), atol=1e-5)
    assert _stepped_solver(_relaxing_group(10).prepare()._problem, 1, 0.5).method_ == 'LSODA'
    heat = HeatDiffusionSystem(shape=(80, 80)).prepare().system_._problem
    banded = _stepped_solver(heat, 0.2, 0.1)
    assert banded.method_ == 'LSODA'
    assert 'lband' in banded.jacobian_options_
    with pytest.raises(SolverError, match='dense 10000 x 10000 matrix'):
        _relaxing_group(10_000).set_params(solver=ScipyODESolver(method='LSODA')).prepare().simulate(until=1, dt=0.5)


def _tub(s):
    """Return the inflow from time 5 and the drain of the water while there is water."""
    water = s.target.water
    return [RateLaw(water.diff(s.t), 5 * step(s.t, 5) - where(water > 0, 5, 0))]


def _tubs(solver='auto'):
    """Return a group of tubs that empty and then rest on their drain switch."""
    return _system('tubs', _tub, size=1, water=[20.0], solver=solver)


def test_default_solver_finishes_a_state_resting_on_a_discontinuity():
    """Test that a tub resting empty on its drain switch keeps the right values, whichever method carries the run."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        trajectory = _tubs().prepare().simulate(until=6, dt=1)
    np.testing.assert_allclose(np.ravel(trajectory.get_values('tubs.water'))[:5], [20, 15, 10, 5, 0], atol=1e-3)


def test_default_solver_leaves_lsoda_when_it_stalls(monkeypatch):
    """Test that a stall moves the default method on to RK45, and raises an error on an explicit LSODA."""
    monkeypatch.setattr(_scipy_ode, 'STALL_STEP', math.inf)
    monkeypatch.setattr(_scipy_ode, 'STALL_STEPS', 1)
    with pytest.warns(UserWarning, match='LSODA stalled'):
        solver = _stepped_solver(_tubs().prepare()._problem, 6, 1)
    assert solver.method_ == 'RK45'
    with pytest.raises(SolverError, match='stalled'):
        _tubs(ScipyODESolver(method='LSODA')).prepare().simulate(until=6, dt=1)


def test_default_solver_starts_from_rest_with_a_vanishing_drift():
    """Test that an oscillator at rest, forced by `sin(t)`, leaves its zero drift at time 0 and follows the solution."""

    def forced(s):
        """Return the motion of the oscillator forced by `sin(t)`."""
        x, v = s.target.x, s.target.v
        return [RateLaw(x.diff(s.t), v), RateLaw(v.diff(s.t), -4 * x + sin(s.t))]

    trajectory = _system('oscillator', forced, x=0.0, v=0.0).prepare().simulate(until=5, dt=0.01)
    times = trajectory.times
    expected = (np.sin(times) - np.sin(2 * times) / 2) / 3
    np.testing.assert_allclose(trajectory.get_values('oscillator.x'), expected, atol=1e-6)


def _decay():
    """Return the system of exponential decay."""
    return System(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=-1.0)],
    )


def test_scipy_ode_keeps_integrator_across_frames():
    """Test that 1000 frames of ScipyODE evaluate the drift at most twice as often as one SciPy integration."""
    problem = _decay().prepare()._problem
    calls = []

    def counted(t, x, p):
        calls.append(t)
        return problem.drift(t, x, p)

    p = problem.initial_parameters
    solver = ScipyODESolver(until=10.0).prepare(dataclasses.replace(problem, drift=counted))
    state = solver.initial_state_
    for frame in range(1000):
        state = solver.run(StepSpec(frame * 0.01, state, p, (frame + 1) * 0.01), final=frame == 999).state_
    frames = len(calls)
    calls.clear()
    solve_ivp(lambda t, x: counted(t, x, p), (0.0, 10.0), problem.initial_state, rtol=1e-6, atol=1e-9)
    assert frames <= 2 * len(calls)
    assert float(problem.states.read(state, 'a.x')) == pytest.approx(np.exp(-10.0), rel=1e-4)


def test_scipy_ode_restarts_on_parameter_change():
    """Test that a decay rate set to zero at time 5 keeps the state constant from then on."""
    schedule = {'g.r': lambda t: -1.0 if t < 5 - 1e-9 else 0.0}
    values = _decay().prepare().simulate(until=10, dt=0.01, schedule=schedule).get_values('a.x')
    assert values[500] == pytest.approx(np.exp(-5.0), rel=1e-5)
    assert np.all(values[500:] == values[500])


def _chatter(s):
    """Return a rate whose sign switches when the water reaches a level."""
    w = s.target.w
    return [RateLaw(w.diff(s.t), 5 - 10 * ceil(w / 100))]


def test_default_solver_names_a_state_sliding_along_a_discontinuity():
    """Test that a state sliding along a switch no event sees ends in a named error instead of a run without end."""
    with pytest.warns(UserWarning, match='LSODA stalled'), pytest.raises(SolverError, match='slides along'):
        _system('tub', _chatter, w=20.0).prepare().simulate(until=6, dt=1)


def test_default_solver_names_a_blow_up():
    """Test that a solution that diverges in finite time ends in an error that names the blow-up."""
    with pytest.raises(SolverError, match='blows up'):
        FiniteTimeBlowUpSystem().prepare().simulate(2, dt=0.01)


def test_default_solver_follows_fast_exponential_growth():
    """Test that a state on a fast manifold, growing exponentially without a blow-up, stays close to its growth rate."""
    system = TakensBogdanovFormSystem(x=5.0, y=5.0, beta1=0.1, beta2=-1.0).prepare()
    values = system.simulate(20, dt=0.1).get_values('form.x')
    assert np.log(values[-1] / values[-11]) == pytest.approx(1.0, rel=0.01)


def test_default_solver_steps_large_diffusive_fields_explicitly():
    """Test that a large field whose frames span a few stable steps runs with RK45, not a costly implicit method."""
    problem = FitzhughNagumoSystem().prepare().system_._problem
    assert _stepped_solver(problem, 4, 2.0).method_ == 'RK45'


def _stiff_fisher_problem():
    """Return Fisher-KPP on 4000 points whose diffusion makes the largest eigenvalue about 1600."""
    return FisherKPPSystem(STIFF_SIZE, D=1.0, spacing=0.05, backend=NumpyBackend()).prepare().system_._problem


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
    stepper = ScipyODESolver(method=method, rtol=1e-6, jacobian=jacobian)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    start = time.perf_counter()
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, until)).state_
    return time.perf_counter() - start, state, calls[0], stepper


def _event_run(jacobian):
    """Return the duration and the event time of stiff Fisher-KPP that stops when its total grows by half."""
    fisher = FisherKPPSystem(STIFF_SIZE, D=1.0, spacing=0.05).prepare().system_
    line = fisher.elements[0]
    level = 1.5 * line.values['u'].sum()
    system = System(
        elements=[line],
        interactions=[*fisher.interactions, LevelInteraction('threshold', line='line', level=level)],
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        backend=NumpyBackend(),
    )
    start = time.perf_counter()
    trajectory = system.prepare().simulate(until=10.0, dt=1.0)
    times = trajectory.times
    return time.perf_counter() - start, times[-1]


def _hutchinson_run(jacobian):
    """Return the duration and the final state of a diffusive Hutchinson equation on 2000 points with BDF."""
    size = 2000
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    system = _system(
        'line',
        _diffusive_hutchinson,
        space=LatticeSpace(size, spacing=0.05),
        u=initial,
        D=1.0,
        r=1.2,
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        backend=NumpyBackend(),
    )
    start = time.perf_counter()
    trajectory = system.prepare().simulate(until=5.0, dt=1.0)
    state = trajectory.states[-1]
    return time.perf_counter() - start, state


class LevelInteraction(Interaction):
    """Threshold interaction."""

    def __init__(
        self,
        label: str,
        *,
        line: Role,
        level: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.line = line
        self.level = level

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event that ends the run when the total of the line reaches the level."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [EventLaw(total(context.line.u) - context.level, terminal=True)]


def _stiff_fisher(size=SIZE):
    """Return Fisher-KPP whose diffusion makes the largest eigenvalue about 1600."""
    return FisherKPPSystem(size, D=1.0, spacing=0.05).prepare().system_._problem


def _advance(problem, until=1.0, p=None, **options):
    """Return the state of a problem after one call of ScipyODE with options, and the stepper."""
    stepper = ScipyODESolver(**options)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    parameters = problem.initial_parameters if p is None else p
    return stepper.run(StepSpec(0.0, problem.initial_state, parameters, until)).state_, stepper


def _relative(first, second):
    """Return the relative difference of two states."""
    return np.linalg.norm(first - second) / np.linalg.norm(second)


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def _event_time(size, jacobian):
    """Return the time of the terminal event of Fisher-KPP when its total grows by half."""
    initial = (np.arange(size) < 0.05 * size).astype(float)
    system = _system(
        'line',
        _fisher_kpp,
        space=LatticeSpace(size, spacing=0.05, boundary={'u': 'zero-flux'}),
        interactions=[LevelInteraction('threshold', line='line', level=1.5 * initial.sum())],
        u=initial,
        D=1.0,
        r=1.0,
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
    )
    trajectory = system.prepare().simulate(until=10, dt=1.0)
    return trajectory.times[-1], ScipyODESolver(method='BDF', jacobian=jacobian, until=10).prepare(trajectory.problem)


def _hutchinson(size, jacobian):
    """Return the final state of a diffusive Hutchinson equation run with BDF."""
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    system = _system(
        'line',
        _diffusive_hutchinson,
        space=LatticeSpace(size, spacing=0.05, boundary={'u': 'zero-flux'}),
        u=initial,
        D=1.0,
        r=1.2,
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
    )
    trajectory = system.prepare().simulate(until=4, dt=1.0)
    return trajectory.states[-1]


@pytest.mark.benchmark
def test_sparse_jacobians_speed_up_stiff_fields(record_property):
    """Test that the pattern, Radau, and the exact Jacobian are far faster than BDF on the dense path."""
    problem = _stiff_fisher_problem()
    _timed_scipy(problem, 'BDF', 'auto', until=0.1)
    pattern, reference, pattern_calls, stepper = _timed_scipy(problem, 'BDF', 'auto')
    dense, dense_state, _, _ = _timed_scipy(problem, 'BDF', 'dense')
    radau, radau_state, _, _ = _timed_scipy(problem, 'Radau', 'auto')
    exact, exact_state, exact_calls, exact_stepper = _timed_scipy(problem, 'BDF', 'exact')
    split = CrankNicolsonSolver(dt=0.01, until=10.0).prepare(problem)
    start = time.perf_counter()
    split.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 10.0), final=True)
    crank_nicolson = time.perf_counter() - start
    for name, value in [('pattern', pattern), ('radau', radau), ('exact', exact), ('crank_nicolson', crank_nicolson)]:
        record_property(f'dense_over_{name}', dense / value)
    assert dense >= 20 * pattern
    assert radau * 10 <= dense
    assert exact * 10 <= dense
    for solved in [dense_state, radau_state, exact_state]:
        assert np.linalg.norm(solved - reference) / np.linalg.norm(reference) < 1e-5
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


@pytest.mark.benchmark
def test_sparse_jacobians_keep_delayed_stiff_fields_accurate(record_property):
    """Test that a delayed stiff field with the pattern agrees with the dense path."""
    pattern, state = _hutchinson_run('auto')
    dense, dense_state = _hutchinson_run('dense')
    record_property('dense_over_pattern', dense / pattern)
    assert np.linalg.norm(state - dense_state) / np.linalg.norm(dense_state) < 1e-5


def test_bdf_receives_the_pattern_and_agrees_with_the_dense_path():
    """Test that BDF receives the pattern by default and agrees with the dense path."""
    problem = _stiff_fisher()
    sparse, stepper = _advance(problem, method='BDF')
    assert stepper.jacobian_options_['jac_sparsity'].nnz == 3 * SIZE - 2
    dense, dense_stepper = _advance(problem, method='BDF', jacobian='dense')
    assert dense_stepper.jacobian_options_ == {}
    assert dense_stepper.jacobian_pattern_ is None
    assert _relative(sparse, dense) < 1e-5


def test_radau_and_exact_agree_with_the_pattern():
    """Test that Radau with the pattern and BDF with the exact Jacobian agree with BDF and the pattern."""
    problem = _stiff_fisher()
    reference, _ = _advance(problem, method='BDF', rtol=1e-8)
    radau, _ = _advance(problem, method='Radau', rtol=1e-8)
    exact, stepper = _advance(problem, method='BDF', rtol=1e-8, jacobian='exact')
    assert stepper.jacobian_options_ == {}
    assert stepper._solver.njev >= 1
    assert _relative(radau, reference) < 1e-5
    assert _relative(exact, reference) < 1e-5


def test_lsoda_receives_half_bandwidths():
    """Test that LSODA receives half bandwidths of 1 for a three-point Laplacian with zero-flux edges."""
    problem = _stiff_fisher()
    banded, stepper = _advance(problem, method='LSODA')
    assert stepper.jacobian_options_ == {'lband': 1, 'uband': 1}
    exact, _ = _advance(problem, method='LSODA', jacobian='exact')
    reference, _ = _advance(problem, method='BDF', rtol=1e-8)
    assert _relative(banded, reference) < 1e-5
    assert _relative(exact, reference) < 1e-5


def test_lsoda_without_narrow_band_receives_nothing():
    """Test that LSODA receives no bandwidths when periodic edges make the band as wide as the line."""
    space = LatticeSpace(50, spacing=0.25, boundary={'u': 'periodic'})
    system = _system('line', _fisher_kpp, space=space, u=np.linspace(0, 1, 50), D=1.0, r=1.0)
    _, stepper = _advance(system.prepare()._problem, method='LSODA')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is not None


def test_dense_pattern_is_not_passed():
    """Test that a pattern denser than one half is not passed."""

    def mixed(s):
        """Return a decay of every member and a gain from the total of the group."""
        x = s.target.x
        return [RateLaw(x.diff(s.t), -x + 0.1 * total(x))]

    system = _system('group', mixed, size=10, x=np.arange(10.0))
    problem = system.prepare()._problem
    _, stepper = _advance(problem, method='BDF')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is None


@pytest.mark.parametrize('jacobian', ['auto', 'exact', 'dense'])
def test_explicit_methods_ignore_the_option(jacobian):
    """Test that RK45 builds no structure and returns the same trajectory with every option."""
    problem = FisherKPPSystem(100).prepare().system_._problem
    state, stepper = _advance(problem, method='RK45', jacobian=jacobian)
    reference, _ = _advance(problem, method='RK45')
    assert stepper.jacobian_pattern_ is None
    assert stepper.jacobian_options_ == {}
    np.testing.assert_array_equal(state, reference)


def test_unknown_option_raises():
    """Test that an unknown Jacobian option or method raises an error that lists the options."""
    problem = FisherKPPSystem(10).prepare().system_._problem
    with pytest.raises(SolverError, match="'auto', 'exact', 'dense'"):
        ScipyODESolver(jacobian='sparse', random_state=0).prepare(problem)
    with pytest.raises(SolverError, match='BDF'):
        ScipyODESolver(method='Euler', random_state=0).prepare(problem)


def test_exact_option_with_callable_law_raises_on_initialize():
    """Test that asking for the exact Jacobian of a callable law raises when initializing."""

    def decay(s):
        """Return the decay of the line as a Python function."""
        return [RateLaw(s.target.u.diff(s.t), _callable_rate)]

    system = _system('line', decay, space=LatticeSpace(10), u=0.0)
    problem = system.prepare()._problem
    with pytest.raises(ValidationError, match="jacobian='auto'"):
        ScipyODESolver(method='BDF', jacobian='exact', random_state=0).prepare(problem)


def test_exact_jacobian_follows_parameter_changes():
    """Test that a parameter change between calls reaches the exact Jacobian of the next call."""
    problem = _stiff_fisher(100)
    p = problem.initial_parameters.copy()
    states = {}
    for option in ['exact', 'dense']:
        stepper = ScipyODESolver(method='BDF', rtol=1e-8, jacobian=option)
        stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
        x = stepper.run(StepSpec(0.0, problem.initial_state, p, 0.5)).state_
        changed = p.copy()
        changed[problem.parameters.get_entry('line.D').indices] = 3.0
        states[option] = stepper.run(StepSpec(0.5, x, changed, 1.0)).state_
        if option == 'exact':
            jacobian = stepper._solver.jac(0.5, x).toarray()
            exact = problem.jacobian.prepare(problem).run(RunStateSpec(0.5, x, changed)).values_.toarray()
            np.testing.assert_allclose(jacobian, exact)
    assert _relative(states['exact'], states['dense']) < 1e-5


def test_event_on_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern finds the event of a stiff field at the time of the dense path."""
    time, solver = _event_time(SIZE, 'auto')
    dense, _ = _event_time(SIZE, 'dense')
    assert 0 < time < 10
    assert abs(time - dense) < 1e-6
    assert 'jac_sparsity' in solver.jacobian_options_


def test_delayed_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern integrates a delayed stiff field like the dense path."""
    assert _relative(_hutchinson(100, 'auto'), _hutchinson(100, 'dense')) < 1e-5


def test_stability_warning_names_the_stiff_path():
    """Test that the stability warning of a fixed-step explicit stepper names BDF and Radau."""
    problem = _stiff_fisher(50)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        RK4Solver(dt=0.01, random_state=0).prepare(problem)


def test_adaptive_explicit_method_warns_once_on_stiff_field():
    """Test that RK45 warns once on the first frame that spans more than 1000 stable steps, and not later."""
    problem = _stiff_fisher(100)
    assert problem.stable_step == pytest.approx(1.25e-3)
    stepper = ScipyODESolver(method='RK45')
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        x = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stepper.run(StepSpec(2.0, x, problem.initial_parameters, 4.0))
