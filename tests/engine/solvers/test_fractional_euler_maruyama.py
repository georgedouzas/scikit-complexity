"""Test the `FractionalEulerMaruyamaSolver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import SolverError
from skcomplex.core.laws import FractionalNoiseLaw, NoiseLaw
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import (
    EulerMaruyamaSolver,
    FractionalEulerMaruyamaSolver,
    HorizonError,
    fractional_gaussian_noise,
)


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
    noise = fractional_gaussian_noise(64, hurst, 10000, np.random.default_rng(0))
    for lag in (0, 1, 2, 10):
        products = noise[: 64 - lag] * noise[lag:]
        expected = 0.5 * (abs(lag + 1) ** (2 * hurst) - 2 * lag ** (2 * hurst) + abs(lag - 1) ** (2 * hurst))
        error = products[0].std() / np.sqrt(products.shape[1])
        assert abs(products[0].mean() - expected) < 3 * error


def test_fractional_euler_maruyama_correlates_consecutive_increments():
    """Test that consecutive increments of 1000 members over 1000 steps correlate as 2^(2H - 1) - 1."""
    trajectory = (
        Simulation(_walkers(0.75), solver=FractionalEulerMaruyamaSolver(dt=1.0), dt=1.0, random_state=0)
        .check()
    )
    increments = np.diff(trajectory.run(1000).trajectory_.values('w.x'), axis=0)
    correlation = np.mean(increments[1:] * increments[:-1]) / np.mean(increments**2)
    assert correlation == pytest.approx(2 ** (2 * 0.75 - 1) - 1, abs=0.02)


def test_fractional_euler_maruyama_gives_superdiffusion():
    """Test that the ensemble mean squared displacement of 1000 members with H = 0.75 scales with exponent 1.5."""
    simulation = (
        Simulation(_walkers(0.75), solver=FractionalEulerMaruyamaSolver(dt=0.01), dt=0.1, random_state=0)
        .check()
    )
    trajectory = simulation.run(10).trajectory_
    squared = np.mean(trajectory.values('w.x')[1:] ** 2, axis=1)
    assert np.polyfit(np.log(trajectory.times[1:]), np.log(squared), 1)[0] == pytest.approx(1.5, abs=0.05)


def test_fractional_euler_maruyama_matches_noise_at_one_half():
    """Test that a Hurst index of 1/2 gives the variance of an ordinary noise law within 2 standard errors."""
    variances = []
    for fractional in (True, False):
        system = _walkers(0.5, members=4000, fractional=fractional)
        solver = FractionalEulerMaruyamaSolver(dt=0.01) if fractional else EulerMaruyamaSolver(dt=0.01)
        final = Simulation(system, solver=solver, dt=0.5, random_state=1).check().run(1).trajectory_.values('w.x')[-1]
        variances.append((final.var(), final.var() * np.sqrt(2 / final.size)))
    (first, first_error), (second, second_error) = variances
    assert abs(first - second) < 2 * np.hypot(first_error, second_error)


def test_fractional_euler_maruyama_takes_the_horizon_from_the_run():
    """Test that a run without an explicit horizon completes, and two runs with random state 0 are equal."""
    runs = [
        (
            Simulation(_walkers(0.3, members=20), solver=FractionalEulerMaruyamaSolver(dt=0.01), dt=0.1, random_state=0)
            .check()
        )
        .run(2)
        .trajectory_.values('w.x')
        for _ in range(2)
    ]
    assert runs[0].shape == (21, 20)
    assert np.array_equal(*runs)


def test_fractional_euler_maruyama_rejects_times_beyond_the_horizon():
    """Test that a frame-by-frame run with horizon 10 extended to time 12 names the horizon and the time."""
    solver = FractionalEulerMaruyamaSolver(dt=0.1, horizon=10)
    simulation = Simulation(_walkers(0.7, members=5), solver=solver, dt=1.0, random_state=0).check().run(10)
    with pytest.raises(HorizonError, match=r'horizon 10, but the run asked for t = 11.0. .*larger horizon'):
        simulation.extend(12)


def test_euler_maruyama_rejects_fractional_noise():
    """Test that the Euler-Maruyama stepper names the fractional Euler-Maruyama stepper for fractional noise."""
    problem = compile_system(_walkers(0.7, members=2))
    with pytest.raises(SolverError, match=r'FractionalEulerMaruyamaSolver'):
        EulerMaruyamaSolver().prepare(problem, np.random.default_rng(0))


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
    simulation = Simulation(System(elements=[walker('w')]), solver=solver, dt=1, random_state=0).check()
    simulation.run(5)
    simulation.update_parameters({'w.H': 0.6})
    with pytest.raises(SolverError, match=r'Hurst index changed'):
        simulation.extend(6)
