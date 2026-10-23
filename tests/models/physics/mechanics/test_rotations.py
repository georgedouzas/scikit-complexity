"""Test the `_rotations` module of `skcomplex.models.physics.mechanics`."""

import numpy as np
import pytest

from skcomplex.models.physics.mechanics import (
    gimballed_gyroscope,
    rotating_balls_on_rods,
    rotating_swing,
)

from ..conftest import _relative_spread, _run, _series


def test_rotating_swing_conserves_angular_momentum():
    """Test that the swing without torque keeps its angular momentum while the chairs swing out."""
    trajectory = _run(rotating_swing(), 10.0, 0.05)
    theta, omega = _series(trajectory, 'swing.theta', 'swing.phi_d1')
    momentum = (10.0 + 8 * (1 + np.sin(theta)) ** 2) * omega
    assert _relative_spread(momentum) < 1e-8
    assert np.ptp(omega) > 0.1


def test_rotating_balls_turn_faster_when_pulled_in():
    """Test that halving the radius of the balls quadruples their angular velocity."""
    radius = {'right.r': lambda t: 1.0 if t < 1 else 0.5}
    trajectory = _run(rotating_balls_on_rods(1.0, r=1.0), 2.0, 0.1, schedule=radius)
    theta = np.unwrap(_series(trajectory, 'right.theta')[0])
    early, late = np.diff(theta[:10]).mean(), np.diff(theta[-5:]).mean()
    assert late / early == pytest.approx(4.0)


def test_gimballed_gyroscope_precesses_at_torque_over_angular_momentum():
    """Test that the mean precession rate is `-torque / (C spin cos beta)`, shrinking as the rotor spins faster."""
    for spin in (25.0, 50.0, 100.0):
        trajectory = _run(gimballed_gyroscope(spin=spin), 10.0, 0.01)
        mean_rate = trajectory['gyroscope.alpha'][-1] / 10.0
        assert mean_rate == pytest.approx(-1.0 / (2.0 * spin * np.cos(0.3)), rel=0.01)


def test_gimballed_gyroscope_conserves_its_cyclic_momenta_and_precesses():
    """Test that the gyroscope keeps the momenta of its outer ring and rotor and precesses under the torque."""
    trajectory = _run(gimballed_gyroscope(), 5.0, 0.05)
    beta, d_alpha, d_gamma = _series(trajectory, 'gyroscope.beta', 'gyroscope.alpha_d1', 'gyroscope.gamma_d1')
    inertia = 1.0 + 1.5 * np.cos(beta) ** 2 + 2.5 * np.sin(beta) ** 2
    outer = inertia * d_alpha + 2.0 * d_gamma * np.sin(beta)
    rotor = 2.0 * (d_alpha * np.sin(beta) + d_gamma)
    assert np.ptp(outer) < 1e-6
    assert np.ptp(rotor) < 1e-6
    assert np.abs(d_alpha).max() > 0.005
    assert np.ptp(beta) < 0.01
