"""Test the nonlinear dynamics models."""

import numpy as np

from skcomplex.models import run


def test_lorenz_attractor_is_bounded_and_visits_both_lobes():
    """Test that the Lorenz model stays bounded for 50 time units and visits both lobes of its attractor."""
    trajectory = run('lorenz').trajectory_
    x, z = trajectory['lorenz.x'], trajectory['lorenz.z']
    assert np.abs(x).max() < 30
    assert z.max() < 60
    late = x[len(x) // 5 :]
    assert (late > 5).any()
    assert (late < -5).any()


def test_van_der_pol_reaches_limit_cycle():
    """Test that the Van der Pol oscillator settles on a cycle of amplitude near 2."""
    x = run('van_der_pol').trajectory_['oscillator.x']
    assert 1.9 < np.abs(x[-200:]).max() < 2.1


def test_henon_map_stays_on_attractor():
    """Test that the Hénon map stays within its attractor's bounds."""
    trajectory = run('henon_map').trajectory_
    assert np.abs(trajectory['map.x'][100:]).max() < 1.5
    assert np.abs(trajectory['map.y'][100:]).max() < 0.5


def test_henon_heiles_conserves_energy():
    """Test that the Hénon-Heiles star keeps its energy within the solver tolerance."""
    trajectory = run('henon_heiles').trajectory_
    x, y, px, py = (trajectory[name] for name in ('star.x', 'star.y', 'star.x_d1', 'star.y_d1'))
    energy = (px**2 + py**2 + x**2 + y**2) / 2 + x**2 * y - y**3 / 3
    assert np.abs(energy - energy[0]).max() < 1e-3
