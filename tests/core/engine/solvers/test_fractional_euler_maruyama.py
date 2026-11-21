"""Test the `FractionalEulerMaruyamaSolver` class."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, NoiseLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, FractionalEulerMaruyamaSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class NoiseInteraction(Interaction):
    """Unit noise of the walkers, fractional with a Hurst index or ordinary without one."""

    def __init__(self, label: str, *, target: Role = 'w', hurst: float | None = None) -> None:
        super().__init__(label)
        self.target = target
        self.hurst = hurst

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of the walkers."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [NoiseLaw(context.target.x, 1, hurst=self.hurst)]


class TaggedNoiseInteraction(NoiseInteraction):
    """Noise that declares its noise law, so its tags name the noise."""

    law_classes = (NoiseLaw,)


def _walkers(hurst, members=1000, fractional=True, solver='auto', noise=NoiseInteraction):
    """Return a system of walkers driven by fractional noise of a Hurst index, or by ordinary noise."""
    walkers = Element('w', values={'x': np.zeros(members)}, size=members)
    interaction = noise('noise', hurst=hurst if fractional else None)
    return System(elements=[walkers], interactions=[interaction], solver=solver)


@pytest.mark.parametrize('hurst', [0.2, 0.5, 0.8])
def test_fractional_gaussian_noise_has_the_covariance_of_fractional_noise(hurst):
    """Test that sample covariances of 10000 paths match the covariance of fractional Gaussian noise."""
    noise = FractionalEulerMaruyamaSolver._get_fractional_gaussian_noise(64, hurst, 10000, np.random.default_rng(0))
    for lag in (0, 1, 2, 10):
        products = noise[: 64 - lag] * noise[lag:]
        expected = 0.5 * (abs(lag + 1) ** (2 * hurst) - 2 * lag ** (2 * hurst) + abs(lag - 1) ** (2 * hurst))
        error = products[0].std() / np.sqrt(products.shape[1])
        assert abs(products[0].mean() - expected) < 3 * error


def test_fractional_euler_maruyama_correlates_consecutive_increments():
    """Test that consecutive increments of 1000 members over 1000 steps correlate as 2^(2H - 1) - 1."""
    system = _walkers(0.75, solver=FractionalEulerMaruyamaSolver(dt=1.0)).prepare()
    trajectory = system.simulate(until=1000, dt=1.0, random_state=0)
    increments = np.diff(trajectory.get_values('w.x'), axis=0)
    correlation = np.mean(increments[1:] * increments[:-1]) / np.mean(increments**2)
    assert correlation == pytest.approx(2 ** (2 * 0.75 - 1) - 1, abs=0.02)


def test_fractional_euler_maruyama_gives_superdiffusion():
    """Test that the ensemble mean squared displacement of 1000 members with H = 0.75 scales with exponent 1.5."""
    system = _walkers(0.75, solver=FractionalEulerMaruyamaSolver(dt=0.01)).prepare()
    trajectory = system.simulate(until=10, dt=0.1, random_state=0)
    squared = np.mean(trajectory.get_values('w.x')[1:] ** 2, axis=1)
    assert np.polyfit(np.log(trajectory.times[1:]), np.log(squared), 1)[0] == pytest.approx(1.5, abs=0.05)


def test_fractional_euler_maruyama_matches_noise_at_one_half():
    """Test that a Hurst index of 1/2 gives the variance of an ordinary noise law within 2 standard errors."""
    variances = []
    for fractional in (True, False):
        solver = FractionalEulerMaruyamaSolver(dt=0.01) if fractional else EulerMaruyamaSolver(dt=0.01)
        system = _walkers(0.5, members=4000, fractional=fractional, solver=solver).prepare()
        final = system.simulate(until=1, dt=0.5, random_state=1).get_values('w.x')[-1]
        variances.append((final.var(), final.var() * np.sqrt(2 / final.size)))
    (first, first_error), (second, second_error) = variances
    assert abs(first - second) < 2 * np.hypot(first_error, second_error)


def test_fractional_euler_maruyama_takes_the_horizon_from_the_run():
    """Test that a run without an explicit horizon completes, and two runs with random state 0 are equal."""
    system = _walkers(0.3, members=20, solver=FractionalEulerMaruyamaSolver(dt=0.01)).prepare()
    runs = [system.simulate(until=2, dt=0.1, random_state=0).get_values('w.x') for _ in range(2)]
    assert runs[0].shape == (21, 20)
    assert np.array_equal(*runs)


@pytest.mark.parametrize(
    'system',
    [_walkers(0.75, members=2), _walkers(0.75, members=2, noise=TaggedNoiseInteraction)],
)
def test_fractional_noise_runs_with_the_fractional_euler_maruyama_by_default(system):
    """Test that a system of fractional noise runs without a solver, whether or not its tags name its laws."""
    trajectory = system.prepare().simulate(until=1, dt=0.5, random_state=0)
    assert trajectory.get_values('w.x').shape == (3, 2)
    assert np.all(trajectory.get_values('w.x')[-1] != 0.0)


def test_euler_maruyama_rejects_fractional_noise():
    """Test that the Euler-Maruyama stepper names the fractional Euler-Maruyama stepper for fractional noise."""
    problem = _walkers(0.7, members=2).prepare()._problem
    with pytest.raises(SolverError, match=r'FractionalEulerMaruyamaSolver'):
        EulerMaruyamaSolver(random_state=0).prepare(problem)


def test_a_simulation_with_the_euler_maruyama_stepper_rejects_fractional_noise():
    """Test that a run with the Euler-Maruyama stepper names the fractional stepper for a Hurst index of 0.7."""
    with pytest.raises(SolverError, match=r'FractionalEulerMaruyamaSolver'):
        _walkers(0.7, members=2, solver=EulerMaruyamaSolver(dt=0.1)).prepare().simulate(until=1, random_state=0)


def test_a_hurst_index_of_one_half_runs_with_the_euler_maruyama_stepper():
    """Test that a noise law with a Hurst index of 1/2 runs with the Euler-Maruyama stepper."""
    system = _walkers(0.5, members=2, solver=EulerMaruyamaSolver(dt=0.1)).prepare()
    trajectory = system.simulate(until=1, random_state=0)
    assert trajectory.get_values('w.x').shape[1] == 2


def test_fractional_euler_maruyama_rejects_a_changed_hurst_index():
    """Test that a change of the Hurst index after the increments were drawn raises an error."""

    noise = Interaction(
        'noise',
        roles={'target': 'w'},
        laws=lambda s: [NoiseLaw(s.target.x, 1, hurst=s.target.H)],
    )
    solver = FractionalEulerMaruyamaSolver(dt=0.1, horizon=10)
    system = System(elements=[Element('w', values={'x': 0.0, 'H': 0.3})], interactions=[noise], solver=solver)
    with pytest.raises(SolverError, match=r'Hurst index changed'):
        system.prepare().simulate(
            until=6,
            dt=1,
            random_state=0,
            schedule={'w.H': lambda t: 0.6 if t >= 5 else 0.3},
        )


def test_fractional_euler_maruyama_solver_get_fractional_gaussian_noise():
    """Test `FractionalEulerMaruyamaSolver._get_fractional_gaussian_noise` on a small case."""
    solver = FractionalEulerMaruyamaSolver
    noise = solver._get_fractional_gaussian_noise(100, 0.75, 2000, np.random.default_rng(0))
    assert noise.shape == (100, 2000)
    assert round(float(np.mean(noise[1:] * noise[:-1])), 1) == 0.4
