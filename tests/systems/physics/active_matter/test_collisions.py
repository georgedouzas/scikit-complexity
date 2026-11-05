"""Test the `_collisions` module of `skcomplex.systems.physics.active_matter`."""

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import describe
from skcomplex.systems.physics.active_matter import binary_collision_flocking

from .conftest import _stationary_polarization


def test_binary_collision_flocking_orders_below_noise_threshold() -> None:
    """Particles flock at low noise and stay disordered above the largest threshold noise of about 0.90."""
    polarizations = []
    for sigma in (0.2, 1.0):
        system = binary_collision_flocking(400, size=20.0, sigma=sigma, random_state=0)
        simulation = (
            Simulation(
                system,
                dt=1,
                random_state=0,
                observables=describe('binary_collision_flocking').observables_of(system),
            )
            .prepare()
            .run(until=200)
        )
        polarizations.append(_stationary_polarization(simulation.trajectory_, 100))
    ordered, disordered = polarizations
    assert ordered > 0.6
    assert disordered < 0.2
