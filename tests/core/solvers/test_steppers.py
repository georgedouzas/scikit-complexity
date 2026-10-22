"""Test the steppers and the solver registry."""

import dataclasses
import time

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ceil, sin
from scipy.integrate import solve_ivp

from skcomplex.core.base import SolverError, role
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Rate, mean, step, where
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, EulerMaruyama, Gillespie, Iteration, ScipyODE, resolve_solver
from skcomplex.core.system import System
from skcomplex.models import create, run
from skcomplex.models.biology.ecology import ExponentialGrowth, Population
from skcomplex.models.biology.epidemiology import Compartment, Recovery
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion
from skcomplex.models.economics.markets import Cobweb, Market
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


def test_stepper_wrong_form_raises_error():
    """Test that a stepper refuses a problem of another form."""
    with pytest.raises(SolverError, match='jump process form'):
        Gillespie().initialize(_growth_problem(), np.random.default_rng(0))


def test_resolve_solver_by_name_and_instance():
    """Test that solvers resolve from names, the default, and instances."""
    problem = _growth_problem()
    assert isinstance(resolve_solver(problem, 'auto'), ScipyODE)
    assert isinstance(resolve_solver(problem, 'scipy'), ScipyODE)
    stepper = ScipyODE()
    assert resolve_solver(problem, stepper) is stepper
    with pytest.raises(SolverError, match='Available names are'):
        resolve_solver(problem, 'gillespie')


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


def test_euler_maruyama_without_noise_matches_ode():
    """Test that zero noise reduces Euler-Maruyama to the deterministic solution."""
    system = System(
        components=[Asset('s', price=1.0)],
        interactions=[GeometricBrownianMotion('gbm', target='s', mu=1.0, sigma=0.0)],
    )
    problem = compile_system(system)
    stepper = EulerMaruyama(dt=1e-4)
    stepper.initialize(problem, np.random.default_rng(0))
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
    assert state[0] == pytest.approx(np.e, rel=1e-3)


def test_euler_maruyama_nonpositive_dt_raises_error():
    """Test that the internal step must be positive."""
    problem = compile_system(
        System(components=[Asset('s')], interactions=[GeometricBrownianMotion('gbm', target='s')]),
    )
    stepper = EulerMaruyama(dt=0.0)
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='positive'):
        stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)


def test_gillespie_conserves_individuals():
    """Test that recoveries move individuals without creating or destroying them."""
    problem = compile_system(
        System(
            components=[Compartment('I', n=50), Compartment('R')],
            interactions=[Recovery('rec', infected='I', recovered='R', gamma=0.5, stochastic=True)],
        ),
    )
    stepper = Gillespie()
    stepper.initialize(problem, np.random.default_rng(1))
    state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
    assert state.sum() == 50
    assert 0 < state[0] < 50
    assert np.all(state == np.round(state))


def test_iteration_applies_one_step_per_unit_time():
    """Test that the discrete stepper applies the map once per integer step."""
    problem = compile_system(
        System(components=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
    )
    stepper = Iteration()
    stepper.initialize(problem, np.random.default_rng(0))
    assert stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0).tolist() == [2.5]


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
