"""Test the `FractionalEulerMaruyamaSolver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import FractionalNoiseLaw, NoiseLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    FractionalEulerMaruyamaSolver,
)
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


def _walkers(hurst, members=1000, fractional=True):
    """Return a system of walkers driven by fractional noise of a Hurst index, or by ordinary noise."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [FractionalNoiseLaw(s.x, 1, hurst=hurst) if fractional else NoiseLaw(s.x, 1)]

    walker = WalkerElement
    return System(elements=[GroupElement('w', element=walker, size=members)])


@pytest.mark.parametrize('hurst', [0.2, 0.5, 0.8])
def test_fractional_gaussian_noise_has_the_covariance_of_fractional_noise(hurst):
    """Test that sample covariances of 10000 paths match the covariance of fractional Gaussian noise."""
    noise = FractionalEulerMaruyamaSolver._fractional_gaussian_noise(64, hurst, 10000, np.random.default_rng(0))
    for lag in (0, 1, 2, 10):
        products = noise[: 64 - lag] * noise[lag:]
        expected = 0.5 * (abs(lag + 1) ** (2 * hurst) - 2 * lag ** (2 * hurst) + abs(lag - 1) ** (2 * hurst))
        error = products[0].std() / np.sqrt(products.shape[1])
        assert abs(products[0].mean() - expected) < 3 * error


def test_fractional_euler_maruyama_correlates_consecutive_increments():
    """Test that consecutive increments of 1000 members over 1000 steps correlate as 2^(2H - 1) - 1."""
    trajectory = simulate(
        _walkers(0.75),
        until=1000,
        solver=FractionalEulerMaruyamaSolver(dt=1.0),
        dt=1.0,
        random_state=0,
    )
    increments = np.diff(trajectory.values('w.x'), axis=0)
    correlation = np.mean(increments[1:] * increments[:-1]) / np.mean(increments**2)
    assert correlation == pytest.approx(2 ** (2 * 0.75 - 1) - 1, abs=0.02)


def test_fractional_euler_maruyama_gives_superdiffusion():
    """Test that the ensemble mean squared displacement of 1000 members with H = 0.75 scales with exponent 1.5."""
    trajectory = simulate(
        _walkers(0.75),
        until=10,
        solver=FractionalEulerMaruyamaSolver(dt=0.01),
        dt=0.1,
        random_state=0,
    )
    squared = np.mean(trajectory.values('w.x')[1:] ** 2, axis=1)
    assert np.polyfit(np.log(trajectory.times[1:]), np.log(squared), 1)[0] == pytest.approx(1.5, abs=0.05)


def test_fractional_euler_maruyama_matches_noise_at_one_half():
    """Test that a Hurst index of 1/2 gives the variance of an ordinary noise law within 2 standard errors."""
    variances = []
    for fractional in (True, False):
        system = _walkers(0.5, members=4000, fractional=fractional)
        solver = FractionalEulerMaruyamaSolver(dt=0.01) if fractional else EulerMaruyamaSolver(dt=0.01)
        final = simulate(system, until=1, solver=solver, dt=0.5, random_state=1).values('w.x')[-1]
        variances.append((final.var(), final.var() * np.sqrt(2 / final.size)))
    (first, first_error), (second, second_error) = variances
    assert abs(first - second) < 2 * np.hypot(first_error, second_error)


def test_fractional_euler_maruyama_takes_the_horizon_from_the_run():
    """Test that a run without an explicit horizon completes, and two runs with random state 0 are equal."""
    runs = [
        simulate(
            _walkers(0.3, members=20),
            until=2,
            solver=FractionalEulerMaruyamaSolver(dt=0.01),
            dt=0.1,
            random_state=0,
        ).values('w.x')
        for _ in range(2)
    ]
    assert runs[0].shape == (21, 20)
    assert np.array_equal(*runs)


def test_euler_maruyama_rejects_fractional_noise():
    """Test that the Euler-Maruyama stepper names the fractional Euler-Maruyama stepper for fractional noise."""
    problem = PassCompiler().prepare(_walkers(0.7, members=2)).run().problem_
    with pytest.raises(SolverError, match=r'FractionalEulerMaruyamaSolver'):
        EulerMaruyamaSolver(random_state=0).prepare(problem)


def test_fractional_euler_maruyama_rejects_a_changed_hurst_index():
    """Test that a change of the Hurst index after the increments were drawn raises an error."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
            H: Annotated[object, ParameterSpec()] = 0.3,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.H = H

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [FractionalNoiseLaw(s.x, 1, hurst=s.H)]

    walker = WalkerElement
    solver = FractionalEulerMaruyamaSolver(dt=0.1, horizon=10)
    with pytest.raises(SolverError, match=r'Hurst index changed'):
        simulate(
            System(elements=[walker('w')]),
            until=6,
            solver=solver,
            dt=1,
            random_state=0,
            schedule={'w.H': lambda t: 0.6 if t >= 5 else 0.3},
        )
