"""Test the `_ratchets` module of `skcomplex.models.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.models.physics.soft_matter import (
    flashing_ratchet,
    tilted_periodic_potential,
)

STANDARD_ERRORS = 3


def _washboard_velocity(f, *, U0, L, kT, gamma, n_points=400):
    """Return the exact mean velocity of an overdamped particle in the tilted potential `U0 cos(2 pi x / L) - f x`."""
    x = np.linspace(0.0, L, n_points, endpoint=False)
    first, second = np.meshgrid(x, x, indexing='ij')

    def potential(position):
        return U0 * np.cos(2 * np.pi * position / L) - f * position

    integral = np.exp((potential(first) - potential(first - second)) / kT).mean() * L**2
    return L * (1 - math.exp(-f * L / kT)) / integral * kT / gamma


@pytest.mark.parametrize('f', [0.5, 2.0])
def test_tilted_periodic_potential_current(f):
    """Test that the mean velocity matches the exact steady current of the tilted periodic potential."""
    system = tilted_periodic_potential(500, f=f, U0=0.5, L=1.0, kT=0.5)
    simulation = (
        Simulation(system, dt=1.0, solver=EulerMaruyamaSolver(dt=0.005), random_state=0).check().prepare().run(until=30)
    )
    positions = simulation.trajectory_.values('particles.x')[..., 0]
    velocities = (positions[-1] - positions[5]) / 25
    error = velocities.std() / math.sqrt(velocities.size)
    exact = _washboard_velocity(f, U0=0.5, L=1.0, kT=0.5, gamma=1.0)
    assert abs(velocities.mean() - exact) < STANDARD_ERRORS * error + 0.02 * exact


def test_flashing_ratchet_current():
    """Test that an asymmetric sawtooth drives a current of the sign of its asymmetry and a symmetric one none."""
    currents = []
    for asymmetry in (0.1, 0.5):
        system = flashing_ratchet(300, asymmetry=asymmetry, rate=1.0, duration=40.0, random_state=1)
        simulation = (
            Simulation(system, dt=1.0, solver=EulerMaruyamaSolver(dt=0.004), random_state=0)
            .check()
            .prepare()
            .run(until=40)
        )
        velocities = simulation.trajectory_.values('particles.x')[-1, :, 0] / 40
        currents.append((velocities.mean(), velocities.std() / math.sqrt(velocities.size)))
    (asymmetric, asymmetric_error), (symmetric, symmetric_error) = currents
    assert np.sign(asymmetric) == np.sign(0.5 - 0.1)
    assert abs(asymmetric) > STANDARD_ERRORS * asymmetric_error
    assert abs(symmetric) < STANDARD_ERRORS * symmetric_error
