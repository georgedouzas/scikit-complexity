"""Test that SDE steppers draw shared and correlated Wiener increments from the run's random stream."""

import math

import numpy as np
import pytest

from skcomplex.core.base import role
from skcomplex.core.components import Element, Wiener
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Noise
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama, Heun
from skcomplex.core.system import System
from skcomplex.models.economics.finance import geometric_brownian_motion

Stock = Element.define('Stock', states={'n': 0.0})
Pair = Element.define(
    'Pair',
    states={'x': 1.0, 'y': 1.0},
    laws=lambda s: [Noise(s.x, 1.0, process='common'), Noise(s.y, 1.0, process='common')],
)
STORED_GEOMETRIC_BROWNIAN_MOTION = [
    100.0,
    104.33582683330506,
    105.59332585811273,
    106.46422136618422,
    102.1583666086639,
    99.91911505743448,
    95.39643613168118,
    92.82823341163548,
    92.42940524533182,
    98.80963167568167,
    94.28348547843957,
]


class Drive(Interaction):
    """Unit noise on two elements from the two processes of a Wiener element."""

    first: str = role()
    second: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise of both elements."""
        return [Noise(s.first.n, 1.0, process=s.noise.dW[0]), Noise(s.second.n, 1.0, process=s.noise.dW[1])]


def _correlated(covariance):
    """Return two stocks driven by a Wiener element of a covariance."""
    return System(
        components=[Stock('a'), Stock('b'), Wiener('market', dimension=2, covariance=covariance)],
        interactions=[Drive('drive', first='a', second='b', noise='market')],
    )


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('stepper', [EulerMaruyama, Heun])
def test_shared_process_keeps_difference(backend, stepper):
    """Test that one process driving two states keeps their difference at every frame, with correlation 1."""
    trajectory = (
        Simulation(System(components=[Pair('pair')]), backend=backend, solver=stepper(dt=0.01), random_state=0, dt=0.1)
        .run(until=5)
        .trajectory_
    )
    np.testing.assert_array_equal(trajectory['pair.x'], trajectory['pair.y'])
    assert np.corrcoef(trajectory['pair.x'], trajectory['pair.y'])[0, 1] == pytest.approx(1.0)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_correlated_increments_reproduce_covariance(backend):
    """Test that the frame increments of two states driven by a Wiener element have its covariance within 0.1."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    trajectory = (
        Simulation(_correlated(covariance), backend=backend, solver=EulerMaruyama(dt=1.0), random_state=0, dt=1.0)
        .run(until=10_000)
        .trajectory_
    )
    increments = np.diff(np.stack([trajectory['a.n'], trajectory['b.n']]), axis=1)
    np.testing.assert_allclose(np.cov(increments), covariance, atol=0.1)


def test_correlated_noise_repeats_with_same_seed():
    """Test that two runs of a correlated Wiener element with random state 0 on the Numba backend are equal."""
    system = _correlated([[1.0, 0.5], [0.5, 1.0]])
    runs = [
        Simulation(system, backend='numba', random_state=0, dt=0.1).run(until=2).trajectory_.states for _ in range(2)
    ]
    np.testing.assert_array_equal(*runs)


def test_perfect_correlation_keeps_states_equal():
    """Test that processes of correlation 1 keep two states with unit noise equal at every frame."""
    trajectory = Simulation(_correlated([[1.0, 1.0], [1.0, 1.0]]), random_state=0, dt=0.1).run(until=2).trajectory_
    np.testing.assert_allclose(trajectory['a.n'], trajectory['b.n'], atol=1e-12)


def test_unnamed_noise_reproduces_stored_trajectory():
    """Test that geometric Brownian motion with random state 0 repeats the trajectory stored before named processes."""
    trajectory = Simulation(geometric_brownian_motion(), random_state=0, dt=0.05).run(until=0.5).trajectory_
    np.testing.assert_array_equal(trajectory['asset.price'], STORED_GEOMETRIC_BROWNIAN_MOTION)


def test_shared_stratonovich_process_has_exact_mean():
    """Test that `dX = s Y o dW`, `dY = s X o dW` on one process has the mean `(X0 + Y0) exp(s^2 / 2)` at time 1."""
    sigma = 0.5
    Cross = Element.define(
        'Cross',
        states={'x': 1.0, 'y': 0.5},
        laws=lambda s: [
            Noise(s.x, sigma * s.y, convention='stratonovich', process='common'),
            Noise(s.y, sigma * s.x, convention='stratonovich', process='common'),
        ],
    )
    system = System(components=[Cross.group('crosses', 4000)])
    state = Simulation(system, solver=EulerMaruyama(dt=0.01), random_state=0, dt=1.0).run(until=1).state_
    totals = state[:4000] + state[4000:]
    error = totals.std() / math.sqrt(totals.size)
    assert abs(totals.mean() - 1.5 * math.exp(sigma**2 / 2)) < 2 * error
