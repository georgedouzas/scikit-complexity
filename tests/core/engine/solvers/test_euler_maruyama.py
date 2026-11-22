"""Test the `EulerMaruyamaSolver` class."""

import math
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, WienerProcessSpec
from skcomplex.core.components.laws import BaseLaw, NoiseLaw
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, HeunSolver, StepSpec
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.economics.finance import (
    AssetElement,
    GeometricBrownianMotionInteraction,
    GeometricBrownianMotionSystem,
)


class DriveInteraction(Interaction):
    """Unit noise on two elements from the two processes of a `WienerProcessSpec`."""

    def __init__(self, label: str, *, first: Role, second: Role, noise: WienerProcessSpec) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the noise of both elements."""
        super()._prepare(context)
        dw = context.noise.dW
        self.laws_: list[BaseLaw] = [
            NoiseLaw(context.first.n, 1.0, process=dw[0]),
            NoiseLaw(context.second.n, 1.0, process=dw[1]),
        ]


def _correlated(covariance: npt.ArrayLike, backend: object) -> System:
    """Return two holdings driven by a `WienerProcessSpec` of a covariance."""
    noise = WienerProcessSpec('market', dimension=2, covariance=covariance)
    return System(
        elements=[Element('a', values={'n': 0.0}), Element('b', values={'n': 0.0})],
        interactions=[DriveInteraction('drive', first='a', second='b', noise=noise)],
        solver=EulerMaruyamaSolver(dt=1.0),
        backend=backend,
    )


def test_euler_maruyama_without_noise_matches_ode():
    """Test that zero noise reduces Euler-Maruyama to the deterministic solution."""
    system = System(
        elements=[AssetElement('s', price=1.0)],
        interactions=[GeometricBrownianMotionInteraction('gbm', target='s', mu=1.0, sigma=0.0)],
    )
    problem = system.prepare()._problem
    stepper = EulerMaruyamaSolver(dt=1e-4)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_
    assert state[0] == pytest.approx(np.e, rel=1e-3)


def test_euler_maruyama_nonpositive_dt_raises_error():
    """Test that the internal step must be positive."""
    system = System(elements=[AssetElement('s')], interactions=[GeometricBrownianMotionInteraction('gbm', target='s')])
    problem = system.prepare()._problem
    stepper = EulerMaruyamaSolver(dt=0.0)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    with pytest.raises(SolverError, match='positive'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))


def test_euler_maruyama_same_seed_on_both_backends():
    """Test that geometric Brownian motion from 0.1 to 0.4 takes three steps on both backends and ends equal."""
    states = []
    for backend in (NumpyBackend(), NumbaBackend()):
        problem = GeometricBrownianMotionSystem(backend=backend).prepare().system_._problem
        rng = np.random.default_rng(0)
        stepper = EulerMaruyamaSolver(dt=0.1)
        stepper.set_params(random_state=rng).prepare(problem)
        states.append(stepper.run(StepSpec(0.1, problem.initial_state, problem.initial_parameters, 0.4)).state_)
        three_steps = np.random.default_rng(0)
        for _ in range(3):
            three_steps.normal(0.0, 0.1, size=1)
        assert rng.bit_generator.state == three_steps.bit_generator.state
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)


def _common_noise(s):
    """Return the noise of both values of the pair from one process."""
    return [NoiseLaw(s.target.x, 1.0, process='common'), NoiseLaw(s.target.y, 1.0, process='common')]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
@pytest.mark.parametrize('stepper', [EulerMaruyamaSolver, HeunSolver])
def test_shared_process_keeps_difference(backend, stepper):
    """Test that one process driving two states keeps their difference at every frame, with correlation 1."""
    system = System(
        elements=[Element('pair', values={'x': 1.0, 'y': 1.0})],
        interactions=[Interaction('noise', roles={'target': 'pair'}, laws=_common_noise)],
        solver=stepper(dt=0.01),
        backend=backend(),
    )
    trajectory = system.prepare().simulate(until=5, random_state=0, dt=0.1)
    np.testing.assert_array_equal(trajectory.get_values('pair.x'), trajectory.get_values('pair.y'))
    assert np.corrcoef(trajectory.get_values('pair.x'), trajectory.get_values('pair.y'))[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
def test_correlated_increments_reproduce_covariance(backend):
    """Test that the frame increments of two states driven by a `WienerProcessSpec` have its covariance within 0.1."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    trajectory = _correlated(covariance, backend()).prepare().simulate(until=10_000, random_state=0, dt=1.0)
    increments = np.diff(np.stack([trajectory.get_values('a.n'), trajectory.get_values('b.n')]), axis=1)
    np.testing.assert_allclose(np.cov(increments), covariance, atol=0.1)


def test_shared_stratonovich_process_has_exact_mean():
    """Test that `dX = s Y o dW`, `dY = s X o dW` on one process has the mean `(X0 + Y0) exp(s^2 / 2)` at time 1."""
    sigma = 0.5

    def cross_noise(s):
        """Return the noise of each value driven by the other one."""
        x, y = s.target.x, s.target.y
        return [
            NoiseLaw(x, sigma * y, convention='stratonovich', process='common'),
            NoiseLaw(y, sigma * x, convention='stratonovich', process='common'),
        ]

    system = System(
        elements=[Element('crosses', values={'x': np.ones(4000), 'y': np.full(4000, 0.5)}, size=4000)],
        interactions=[Interaction('noise', roles={'target': 'crosses'}, laws=cross_noise)],
        solver=EulerMaruyamaSolver(dt=0.01),
    )
    state = system.prepare().simulate(until=1, random_state=0, dt=1.0).states[-1]
    totals = state[:4000] + state[4000:]
    error = totals.std() / math.sqrt(totals.size)
    assert abs(totals.mean() - 1.5 * math.exp(sigma**2 / 2)) < 2 * error
