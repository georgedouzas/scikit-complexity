"""Test the `_billiards` module of `skcomplex.systems.physics.nonlinear_oscillators`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.physics.nonlinear_oscillators import BilliardsSystem

from ..conftest import _run, _series


@pytest.mark.parametrize('shape', ['sinai', 'mushroom'])
def test_billiards_keep_the_speed_and_the_particle_on_the_table(shape):
    """Test that the billiard particle keeps its speed and never leaves the table."""
    trajectory = _run(BilliardsSystem(shape), 30.0, 0.05, solver=ScipyODESolver())
    x, y, vx, vy = _series(trajectory, 'particle.x', 'particle.y', 'particle.vx', 'particle.vy')
    assert len(trajectory.events) > 10
    assert np.ptp(vx**2 + vy**2) < 1e-12
    if shape == 'sinai':
        assert np.abs(x).max() <= 1 + 1e-6
        assert np.abs(y).max() <= 1 + 1e-6
        assert np.hypot(x, y).min() >= 0.4 - 1e-6
        return
    stem = y < 0
    assert np.hypot(x, y)[~stem].max() <= 1 + 1e-6
    assert np.abs(x[stem]).max(initial=0.0) <= 0.4 + 1e-6
    assert y.min() >= -1 - 1e-6


def _scatter(phi):
    """Return the trajectory of a particle that comes from far away towards the origin from the polar angle `phi`."""
    system = BilliardsSystem(
        'three_disk',
        x=8 * math.cos(phi),
        y=8 * math.sin(phi),
        vx=-math.cos(phi),
        vy=-math.sin(phi),
    )
    return _run(system, 25.0, 0.05, solver=ScipyODESolver())


def test_three_disk_billiard_keeps_the_particle_outside_the_discs_and_lets_it_leave():
    """Test that the scattered particle keeps its speed, never enters a disc, and leaves the scattering region."""
    trajectory = _scatter(1.49)
    x, y, vx, vy = _series(trajectory, 'particle.x', 'particle.y', 'particle.vx', 'particle.vy')
    distance = 2.4 / math.sqrt(3)
    for angle in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6):
        assert np.hypot(x - distance * math.cos(angle), y - distance * math.sin(angle)).min() >= 0.4 - 1e-6
    assert np.ptp(vx**2 + vy**2) < 1e-12
    assert len(trajectory.events) == 3
    assert np.hypot(x[-1], y[-1]) > 8


def test_three_disk_billiard_scatters_back_a_head_on_particle():
    """Test that a particle aimed at the center of a disc through the gap bounces once and goes straight back."""
    trajectory = _scatter(math.pi / 2)
    vx, vy = _series(trajectory, 'particle.vx', 'particle.vy')
    assert len(trajectory.events) == 1
    assert (vx[-1], vy[-1]) == pytest.approx((0.0, 1.0), abs=1e-9)
