"""Test the steppers and the solver registry."""

import numpy as np
import pytest

from skcomplex.compile import compile_system
from skcomplex.core import SolverError, role
from skcomplex.model import Interaction, Rate, System
from skcomplex.models.ecology import ExponentialGrowth, Population
from skcomplex.models.economics import Cobweb, Market
from skcomplex.models.epidemiology import Compartment, Recovery
from skcomplex.models.finance import Asset, GeometricBrownianMotion
from skcomplex.solvers import EulerMaruyama, Gillespie, Iteration, ScipyODE, resolve_solver


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1.0)])
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
        System(elements=[Population('a', n=1)], interactions=[Explosion('boom', target='a')]),
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


def test_euler_maruyama_without_noise_matches_ode():
    """Test that zero noise reduces Euler-Maruyama to the deterministic solution."""
    system = System(
        elements=[Asset('s', price=1.0)],
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
        System(elements=[Asset('s')], interactions=[GeometricBrownianMotion('gbm', target='s')]),
    )
    stepper = EulerMaruyama(dt=0.0)
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='positive'):
        stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)


def test_gillespie_conserves_individuals():
    """Test that recoveries move individuals without creating or destroying them."""
    problem = compile_system(
        System(
            elements=[Compartment('I', n=50), Compartment('R')],
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
        System(elements=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
    )
    stepper = Iteration()
    stepper.initialize(problem, np.random.default_rng(0))
    assert stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0).tolist() == [2.5]
