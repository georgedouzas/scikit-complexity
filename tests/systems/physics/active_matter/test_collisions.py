"""Test the `_collisions` module of `skcomplex.systems.physics.active_matter`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.active_matter import BinaryCollisionFlockingSystem

from .conftest import _stationary_polarization


def test_binary_collision_flocking_orders_below_noise_threshold() -> None:
    """Particles flock at low noise and stay disordered above the largest threshold noise of about 0.90."""
    polarizations = []
    for sigma in (0.2, 1.0):
        system = BinaryCollisionFlockingSystem(400, size=20.0, sigma=sigma, random_state=0).prepare().system_
        trajectory = simulate(
            system,
            until=200,
            dt=1,
            random_state=0,
            observables=BinaryCollisionFlockingSystem.observables,
        )
        polarizations.append(_stationary_polarization(trajectory, 100))
    ordered, disordered = polarizations
    assert ordered > 0.6
    assert disordered < 0.2
