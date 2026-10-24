"""Test the integrators of ordinary differential equations."""

import dataclasses
import time
import warnings
from dataclasses import replace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ceil, sin
from scipy.integrate import solve_ivp

from skcomplex.core.base import Domain, SolverError, ValidationError, parameter, role
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Event, Rate, delayed, mean, step, total, where
from skcomplex.core.space import ContinuousSpace, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import RealTime, Simulation
from skcomplex.engine.solvers import RK4, ScipyODE, VelocityVerlet
from skcomplex.formalisms.fields import CrankNicolson, Field, laplacian
from skcomplex.formalisms.particles import PairPotential, Particle
from skcomplex.models import create, run
from skcomplex.models.biology.ecology import ExponentialGrowth, FisherKPP, Population, fisher_kpp
from skcomplex.models.physics.heat import heat_diffusion
from skcomplex.models.physics.mechanics import harmonic_oscillator


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(components=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1.0)])
    return compile_system(system)


@pytest.mark.parametrize('method', ['RK45', 'DOP853', 'LSODA'])
def test_scipy_ode_matches_exponential(method):
    """Test that the ODE stepper integrates exponential growth accurately."""
    problem = _growth_problem()
    stepper = ScipyODE(method=method, rtol=1e-9, atol=1e-12)
    stepper.initialize(problem, np.random.default_rng(0))
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0)
    assert state[0] == pytest.approx(np.exp(2.0), rel=1e-6)


class Explosion(Interaction):
    """Growth `dn/dt = n**2`, which blows up in finite time."""

    target: str = role()

    def laws(self, s):
        """Return the rate `n**2`."""
        return [Rate(s.target.n, s.target.n**2)]


def test_scipy_ode_failure_raises_error():
    """Test that an integrator failure raises a named error."""
    problem = compile_system(
        System(components=[Population('a', n=1)], interactions=[Explosion('boom', target='a')]),
    )
    stepper = ScipyODE()
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='failed'):
        stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0)


def test_simulation_runs_stiff_model_with_default_solver():
    """Test that a stiff model built directly runs fast with the default solver and matches its registered solver."""
    start = time.perf_counter()
    default = Simulation(create('nanoparticle_corona')).run(until=10800)
    assert time.perf_counter() - start < 5
    registered = run('nanoparticle_corona', settings={'until': 10800})
    np.testing.assert_allclose(default.state_, registered.state_, rtol=1e-4, atol=1e-12)


def test_simulation_default_solver_keeps_accuracy_of_non_stiff_model():
    """Test that a harmonic oscillator run with the default solver matches the analytic solution within 1e-5."""
    simulation = Simulation(harmonic_oscillator()).run(until=10)
    np.testing.assert_allclose(simulation.state_, [np.cos(10), -np.sin(10)], rtol=1e-5)


def _relaxing_group(size):
    """Return a group whose members relax to their mean, so every member reads every other one."""
    Relax = Element.define('Relax', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -(s.x - mean(s.x)))])
    return System(components=[Relax.group('cells', size, x=np.linspace(0.0, 1.0, size))])


def test_default_solver_leaves_lsoda_for_states_its_dense_matrix_cannot_hold():
    """Test that 10^4 coupled states leave LSODA, a banded field of 6400 states keeps it, and LSODA names the limit."""
    simulation = Simulation(_relaxing_group(10_000), dt=0.5).run(until=1)
    assert simulation.stepper_.method_ != 'LSODA'
    start = np.linspace(0.0, 1.0, 10_000)
    np.testing.assert_allclose(simulation.state_, 0.5 + (start - 0.5) * np.exp(-1), atol=1e-5)
    assert Simulation(_relaxing_group(10), dt=0.5).run(until=1).stepper_.method_ == 'LSODA'
    banded = Simulation(heat_diffusion(shape=(80, 80)), dt=0.1).run(until=0.2)
    assert banded.stepper_.method_ == 'LSODA'
    assert 'lband' in banded.stepper_.jacobian_options_
    with pytest.raises(SolverError, match='dense 10000 x 10000 matrix'):
        Simulation(_relaxing_group(10_000), dt=0.5, solver=ScipyODE(method='LSODA')).run(until=1)


Tub = Element.define(
    'Tub',
    states={'water': 20.0},
    laws=lambda s: [Rate(s.water, 5 * step(s.t, 5) - where(s.water > 0, 5, 0))],
)


def test_default_solver_leaves_lsoda_when_a_state_rests_on_a_discontinuity():
    """Test that a group of tubs resting empty on their drain switch finishes with RK45, and explicit LSODA stalls."""
    tubs = System(components=[Tub.group('tubs', 1, water=[20.0])])
    with pytest.warns(UserWarning, match='LSODA stalled'):
        simulation = Simulation(tubs, dt=1).run(until=6)
    assert simulation.stepper_.method_ == 'RK45'
    np.testing.assert_allclose(np.ravel(simulation.trajectory_['tubs.water'])[:5], [20, 15, 10, 5, 0], atol=1e-6)
    with pytest.raises(SolverError, match='stalled'):
        Simulation(tubs, dt=1, solver=ScipyODE(method='LSODA')).run(until=6)


def test_default_solver_starts_from_rest_with_a_vanishing_drift():
    """Test that an oscillator at rest, forced by `sin(t)`, leaves its zero drift at time 0 and follows the solution."""
    Forced = Element.define(
        'Forced',
        states={'x': 0.0, 'v': 0.0},
        laws=lambda s: [Rate(s.x, s.v), Rate(s.v, -4 * s.x + sin(s.t))],
    )
    simulation = Simulation(System(components=[Forced('oscillator')]), dt=0.01).run(until=5)
    times = simulation.trajectory_.times
    expected = (np.sin(times) - np.sin(2 * times) / 2) / 3
    np.testing.assert_allclose(simulation.trajectory_['oscillator.x'], expected, atol=1e-6)


def _decay():
    """Return the system of exponential decay."""
    return System(components=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=-1.0)])


def test_scipy_ode_keeps_integrator_across_frames():
    """Test that 1000 frames of ScipyODE evaluate the drift at most twice as often as one SciPy integration."""
    problem = compile_system(_decay())
    calls = []

    def counted(t, x, p):
        calls.append(t)
        return problem.drift(t, x, p)

    trajectory = Simulation(dataclasses.replace(problem, drift=counted), dt=0.01).run(until=10).trajectory_
    frames = len(calls)
    calls.clear()
    p = problem.initial_parameters
    solve_ivp(lambda t, x: counted(t, x, p), (0.0, 10.0), problem.initial_state, rtol=1e-6, atol=1e-9)
    assert len(trajectory) == 1001
    assert frames <= 2 * len(calls)
    assert trajectory['a.n'][-1] == pytest.approx(np.exp(-10.0), rel=1e-4)


def test_scipy_ode_restarts_on_parameter_change():
    """Test that a decay rate set to zero at time 5 keeps the state constant from then on."""
    simulation = Simulation(_decay(), dt=0.01).run(until=5)
    simulation.update_parameters({'g.r': 0.0})
    values = simulation.extend(until=10).trajectory_['a.n']
    assert values[500] == pytest.approx(np.exp(-5.0), rel=1e-5)
    assert np.all(values[500:] == values[500])


Chatter = Element.define('Chatter', states={'w': 20.0}, laws=lambda s: [Rate(s.w, 5 - 10 * ceil(s.w / 100))])


def test_default_solver_names_a_state_sliding_along_a_discontinuity():
    """Test that a state sliding along a switch no event sees ends in a named error instead of a run without end."""
    with pytest.warns(UserWarning, match='LSODA stalled'), pytest.raises(SolverError, match='slides along'):
        Simulation(System(components=[Chatter('tub')]), dt=1).run(until=6)


def test_default_solver_names_a_blow_up():
    """Test that a solution that blows up in finite time ends in an error that names the blow-up."""
    with pytest.raises(SolverError, match='blows up'):
        Simulation(create('takens_bogdanov_form', x=5.0, y=5.0, beta1=0.1, beta2=-1.0), dt=0.1).run(until=50)


def test_default_solver_steps_large_diffusive_fields_explicitly():
    """Test that a large field whose frames span a few stable steps runs with RK45, not a costly implicit method."""
    simulation = Simulation(create('fitzhugh_nagumo'), dt=2.0).run(until=4)
    assert simulation.stepper_.method_ == 'RK45'


def test_fixed_step_integrators_step_over_switches_of_the_drift():
    """Test that RK4 steps over the swing-up controller switch instead of firing its hidden event forever."""
    system = create('pendulum_swing_up')
    assert any(event.hidden for event in compile_system(system).events)
    start = time.perf_counter()
    simulation = Simulation(system, dt=0.1, solver=RK4(dt=0.01)).run(until=75)
    assert time.perf_counter() - start < 30
    assert simulation.time_ == pytest.approx(75)


class SoftSphere(PairPotential):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    k: float = parameter(100.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


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


SIZE = 400


Threshold = Interaction.define(
    'Threshold',
    roles=['line'],
    parameters={'level': 0.0},
    laws=lambda s: [Event(total(s.line.u) - s.level, terminal=True)],
)


UnitRateHutchinson = Element.define(
    'UnitRateHutchinson',
    states={'u': 0.0},
    parameters={'D': 1.0, 'r': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - delayed(s.u, 1.0)))],
)


def _stiff_fisher(size=SIZE):
    """Return Fisher-KPP whose diffusion makes the largest eigenvalue about 1600."""
    return compile_system(fisher_kpp(size, D=1.0, spacing=0.05))


def _advance(problem, until=1.0, p=None, **options):
    """Return the state of a problem after one call of ScipyODE with options, and the stepper."""
    stepper = ScipyODE(**options)
    stepper.initialize(problem, np.random.default_rng(0))
    parameters = problem.initial_parameters if p is None else p
    return stepper.advance(0.0, problem.initial_state, parameters, until), stepper


def _relative(first, second):
    """Return the relative difference of two states."""
    return np.linalg.norm(first - second) / np.linalg.norm(second)


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def _event_time(size, jacobian):
    """Return the time of the terminal event of Fisher-KPP when its total grows by half."""
    initial = (np.arange(size) < 0.05 * size).astype(float)
    space = LatticeSpace(size, spacing=0.05)
    field = Field('line', FisherKPP, space, boundary={'u': 'zero-flux'}, values={'u': initial})
    simulation = Simulation(
        System(components=[field], interactions=[Threshold('threshold', line='line', level=1.5 * initial.sum())]),
        solver=ScipyODE(method='BDF', jacobian=jacobian),
        dt=1.0,
    )
    return simulation.run(until=10).trajectory_.times[-1], simulation


def _hutchinson(size, jacobian):
    """Return the final state of a diffusive Hutchinson equation run with BDF."""
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    field = Field(
        'line',
        UnitRateHutchinson,
        LatticeSpace(size, spacing=0.05),
        boundary={'u': 'zero-flux'},
        values={'u': initial, 'r': 1.2},
    )
    simulation = Simulation(System(components=[field]), solver=ScipyODE(method='BDF', jacobian=jacobian), dt=1.0)
    return simulation.run(until=4).trajectory_.states[-1]


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
    system = System(
        components=[
            Field(
                'line',
                FisherKPP,
                LatticeSpace(50, spacing=0.25),
                boundary={'u': 'periodic'},
                values={'u': np.linspace(0, 1, 50)},
            ),
        ],
    )
    _, stepper = _advance(compile_system(system), method='LSODA')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is not None


def test_dense_pattern_is_not_passed():
    """Test that a pattern denser than one half is not passed."""
    Member = Element.define('Mixed', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -s.x + 0.1 * total(s.x))])
    problem = compile_system(System(components=[Member.group('group', 10, x=np.arange(10.0))]))
    _, stepper = _advance(problem, method='BDF')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is None


@pytest.mark.parametrize('jacobian', ['auto', 'exact', 'dense'])
def test_explicit_methods_ignore_the_option(jacobian):
    """Test that RK45 builds no structure and returns the same trajectory with every option."""
    problem = compile_system(fisher_kpp(100))
    state, stepper = _advance(problem, method='RK45', jacobian=jacobian)
    reference, _ = _advance(problem, method='RK45')
    assert stepper.jacobian_pattern_ is None
    assert stepper.jacobian_options_ == {}
    np.testing.assert_array_equal(state, reference)


def test_unknown_option_raises():
    """Test that an unknown Jacobian option or method raises an error that lists the options."""
    problem = compile_system(fisher_kpp(10))
    with pytest.raises(SolverError, match="'auto', 'exact', 'dense'"):
        ScipyODE(jacobian='sparse').initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='BDF'):
        ScipyODE(method='Euler').initialize(problem, np.random.default_rng(0))


def test_exact_option_with_callable_law_raises_on_initialize():
    """Test that asking for the exact Jacobian of a callable law raises when initializing."""
    Decay = Element.define('Decay', states={'u': 0.0}, laws=lambda s: [Rate(s.u, _callable_rate)])
    problem = compile_system(System(components=[Field('line', Decay, LatticeSpace(10))]))
    with pytest.raises(ValidationError, match="jacobian='auto'"):
        ScipyODE(method='BDF', jacobian='exact').initialize(problem, np.random.default_rng(0))


def test_exact_jacobian_follows_parameter_changes():
    """Test that a parameter change between calls reaches the exact Jacobian of the next call."""
    problem = _stiff_fisher(100)
    p = problem.initial_parameters.copy()
    states = {}
    for option in ['exact', 'dense']:
        stepper = ScipyODE(method='BDF', rtol=1e-8, jacobian=option)
        stepper.initialize(problem, np.random.default_rng(0))
        x = stepper.advance(0.0, problem.initial_state, p, 0.5)
        changed = p.copy()
        changed[problem.parameters.entry('line.D').indices] = 3.0
        states[option] = stepper.advance(0.5, x, changed, 1.0)
        if option == 'exact':
            jacobian = stepper._solver.jac(0.5, x).toarray()
            np.testing.assert_allclose(jacobian, problem.jacobian(0.5, x, changed).toarray())
    assert _relative(states['exact'], states['dense']) < 1e-5


def test_event_on_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern finds the event of a stiff field at the time of the dense path."""
    time, simulation = _event_time(SIZE, 'auto')
    dense, _ = _event_time(SIZE, 'dense')
    assert 0 < time < 10
    assert abs(time - dense) < 1e-6
    assert 'jac_sparsity' in simulation.stepper_.jacobian_options_


def test_delayed_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern integrates a delayed stiff field like the dense path."""
    assert _relative(_hutchinson(100, 'auto'), _hutchinson(100, 'dense')) < 1e-5


def test_stability_warning_names_the_stiff_path():
    """Test that the stability warning of a fixed-step explicit stepper names BDF and Radau."""
    problem = _stiff_fisher(50)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        RK4(dt=0.01).initialize(problem, np.random.default_rng(0))


def test_adaptive_explicit_method_warns_once_on_stiff_field():
    """Test that RK45 warns once on the first frame that spans more than 1000 stable steps, and not later."""
    problem = _stiff_fisher(100)
    assert problem.stable_step == pytest.approx(1.25e-3)
    stepper = ScipyODE(method='RK45')
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        x = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stepper.advance(2.0, x, problem.initial_parameters, 4.0)
