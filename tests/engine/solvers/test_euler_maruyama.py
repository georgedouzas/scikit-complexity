"""Test the `EulerMaruyamaSolver` class."""

import math
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import NoiseLaw
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver
from skcomplex.errors import SolverError
from skcomplex.models.economics.finance import (
    AssetElement,
    GeometricBrownianMotionInteraction,
    geometric_brownian_motion,
)
from skcomplex.specs import StateSpec, StepSpec, VisualSpec

from ...conftest import _correlated


def test_euler_maruyama_without_noise_matches_ode():
    """Test that zero noise reduces Euler-Maruyama to the deterministic solution."""
    system = System(
        elements=[AssetElement('s', price=1.0)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='s', mu=1.0, sigma=0.0)],
    )
    problem = compile_system(system)
    stepper = EulerMaruyamaSolver(dt=1e-4)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))
    assert state[0] == pytest.approx(np.e, rel=1e-3)


def test_euler_maruyama_nonpositive_dt_raises_error():
    """Test that the internal step must be positive."""
    problem = compile_system(
        System(elements=[AssetElement('s')], interactions=[GeometricBrownianMotionInteraction('gbm', target='s')]),
    )
    stepper = EulerMaruyamaSolver(dt=0.0)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    with pytest.raises(SolverError, match='positive'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))


def test_euler_maruyama_same_seed_on_both_backends():
    """Test that geometric Brownian motion from 0.1 to 0.4 takes three steps on both backends and ends equal."""
    states = []
    for backend in (NumpyBackend(), NumbaBackend()):
        problem = compile_system(geometric_brownian_motion(), backend=backend)
        rng = np.random.default_rng(0)
        stepper = EulerMaruyamaSolver(dt=0.1)
        stepper.set_params(random_state=rng).check().prepare(problem)
        states.append(stepper.run(StepSpec(0.1, problem.initial_state, problem.initial_parameters, 0.4)))
        three_steps = np.random.default_rng(0)
        for _ in range(3):
            three_steps.normal(0.0, 0.1, size=1)
        assert rng.bit_generator.state == three_steps.bit_generator.state
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)


class PairElement(Element):
    """Pair element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y

    def _get_laws(self, s):
        """Return the laws of the pair element."""
        return [NoiseLaw(s.x, 1.0, process='common'), NoiseLaw(s.y, 1.0, process='common')]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
@pytest.mark.parametrize('stepper', [EulerMaruyamaSolver, HeunSolver])
def test_shared_process_keeps_difference(backend, stepper):
    """Test that one process driving two states keeps their difference at every frame, with correlation 1."""
    trajectory = (
        Simulation(
            System(elements=[PairElement('pair')]),
            backend=backend(),
            solver=stepper(dt=0.01),
            random_state=0,
            dt=0.1,
        ).check()
        .run(until=5)
        .trajectory_
    )
    np.testing.assert_array_equal(trajectory.values('pair.x'), trajectory.values('pair.y'))
    assert np.corrcoef(trajectory.values('pair.x'), trajectory.values('pair.y'))[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
def test_correlated_increments_reproduce_covariance(backend):
    """Test that the frame increments of two states driven by a `WienerProcessSpec` have its covariance within 0.1."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    trajectory = (
        Simulation(
            _correlated(covariance), backend=backend(), solver=EulerMaruyamaSolver(dt=1.0), random_state=0, dt=1.0,
        ).check()
        .run(until=10_000)
        .trajectory_
    )
    increments = np.diff(np.stack([trajectory.values('a.n'), trajectory.values('b.n')]), axis=1)
    np.testing.assert_allclose(np.cov(increments), covariance, atol=0.1)


def test_shared_stratonovich_process_has_exact_mean():
    """Test that `dX = s Y o dW`, `dY = s X o dW` on one process has the mean `(X0 + Y0) exp(s^2 / 2)` at time 1."""
    sigma = 0.5

    class CrossElement(Element):
        """Cross element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
            y: Annotated[object, StateSpec()] = 0.5,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the cross element."""
            return [
                NoiseLaw(s.x, sigma * s.y, convention='stratonovich', process='common'),
                NoiseLaw(s.y, sigma * s.x, convention='stratonovich', process='common'),
            ]

    system = System(elements=[GroupElement('crosses', element=CrossElement, size=4000)])
    state = Simulation(system, solver=EulerMaruyamaSolver(dt=0.01), random_state=0, dt=1.0).check().run(until=1).state_
    totals = state[:4000] + state[4000:]
    error = totals.std() / math.sqrt(totals.size)
    assert abs(totals.mean() - 1.5 * math.exp(sigma**2 / 2)) < 2 * error
