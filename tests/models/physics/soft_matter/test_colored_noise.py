"""Test the `_colored_noise` module of `skcomplex.models.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.models.physics.soft_matter import (
    colored_noise_langevin,
)


@pytest.mark.parametrize('tau', [0.1, 2.0])
def test_colored_noise_langevin_velocity_variance(tau):
    """Test that the stationary velocity variance is D / (gamma (1 + gamma tau)) and the noise decays in tau."""
    system = colored_noise_langevin(500, gamma=1.0, D=1.0, tau=tau)
    solver = EulerMaruyama(dt=min(0.01, tau / 50))
    trajectory = Simulation(system, dt=0.5, solver=solver, random_state=0).run(until=40).trajectory_
    velocities, noise = trajectory['particles.v'][20:], trajectory['particles.xi'][20:]
    assert np.mean(velocities**2) == pytest.approx(1 / (1 + tau), rel=0.05)
    assert np.mean(noise**2) == pytest.approx(1 / tau, rel=0.05)
    shift = round(tau / 0.5) or 1
    correlation = np.mean(noise[:-shift] * noise[shift:]) / np.mean(noise**2)
    assert correlation == pytest.approx(math.exp(-shift * 0.5 / tau), abs=0.05)


def test_colored_noise_langevin_white_noise_limit():
    """Test that a short correlation time gives the white noise variance D / gamma."""
    system = colored_noise_langevin(1000, gamma=1.0, D=1.0, tau=0.02)
    simulation = Simulation(system, dt=0.5, solver=EulerMaruyama(dt=0.002), random_state=0).run(until=15)
    assert np.mean(simulation.trajectory_['particles.v'][10:] ** 2) == pytest.approx(1.0, rel=0.05)
