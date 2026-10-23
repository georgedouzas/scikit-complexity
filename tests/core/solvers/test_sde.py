"""Test the fixed-step integrators of stochastic differential equations."""

import numpy as np
import pytest

from skcomplex.core.base import SolverError
from skcomplex.core.compile import compile_system
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.system import System
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion, geometric_brownian_motion


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


def test_euler_maruyama_same_seed_on_both_backends():
    """Test that geometric Brownian motion from 0.1 to 0.4 takes three steps on both backends and ends equal."""
    states = []
    for backend in ('numpy', 'numba'):
        problem = compile_system(geometric_brownian_motion(), backend=backend)
        rng = np.random.default_rng(0)
        stepper = EulerMaruyama(dt=0.1)
        stepper.initialize(problem, rng)
        states.append(stepper.advance(0.1, problem.initial_state, problem.initial_parameters, 0.4))
        three_steps = np.random.default_rng(0)
        for _ in range(3):
            three_steps.normal(0.0, 0.1, size=1)
        assert rng.bit_generator.state == three_steps.bit_generator.state
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)
