"""Test the `_colored_noise` module of `skcomplex.systems.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems.physics.soft_matter import ColoredNoiseLangevinSystem


@pytest.mark.parametrize('tau', [0.1, 2.0])
def test_colored_noise_langevin_velocity_variance(tau):
    """Test that the stationary velocity variance is D / (gamma (1 + gamma tau)) and the noise decays in tau."""
    system = ColoredNoiseLangevinSystem(500, gamma=1.0, D=1.0, tau=tau)
    solver = EulerMaruyamaSolver(dt=min(0.01, tau / 50))
    trajectory = simulate(system, until=40, dt=0.5, solver=solver, random_state=0)
    velocities, noise = trajectory.get_values('particles.v')[20:], trajectory.get_values('particles.xi')[20:]
    assert np.mean(velocities**2) == pytest.approx(1 / (1 + tau), rel=0.05)
    assert np.mean(noise**2) == pytest.approx(1 / tau, rel=0.05)
    shift = round(tau / 0.5) or 1
    correlation = np.mean(noise[:-shift] * noise[shift:]) / np.mean(noise**2)
    assert correlation == pytest.approx(math.exp(-shift * 0.5 / tau), abs=0.05)


def test_colored_noise_langevin_white_noise_limit():
    """Test that a short correlation time gives the white noise variance D / gamma."""
    system = ColoredNoiseLangevinSystem(1000, gamma=1.0, D=1.0, tau=0.02)
    trajectory = simulate(system, until=15, dt=0.5, solver=EulerMaruyamaSolver(dt=0.002), random_state=0)
    assert np.mean(trajectory.get_values('particles.v')[10:] ** 2) == pytest.approx(1.0, rel=0.05)
