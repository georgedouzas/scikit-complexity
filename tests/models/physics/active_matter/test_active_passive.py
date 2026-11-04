"""Test the `_active_passive` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models import run
from skcomplex.models.physics.active_matter import active_passive_flocking


def test_active_passive_flocking_flocks_when_every_agent_is_active():
    """Test that the polarization of the velocities rises above 0.9 with every agent active and flocking on."""
    settings = {'until': 100.0, 'dt': 1.0, 'random_state': 1}
    polarization = run(
        'active_passive_flocking',
        active_share=1.0,
        packing_fraction=0.3,
        alpha=1.0,
        noise=0.1,
        settings=settings,
    ).trajectory_.values('polarization')
    assert polarization[0] < 0.3
    assert polarization[-20:].mean() > 0.9


def test_active_passive_flocking_slows_passive_agents():
    """Test that passive agents without contact lose their speed at the rate of the propulsion."""
    system = active_passive_flocking(4, active_share=0.25, alpha=0.0, noise=0.0, packing_fraction=0.001)
    trajectory = Simulation(system, dt=1.0, random_state=0).check().prepare().run(until=10.0).trajectory_
    speeds = np.linalg.norm(trajectory.values('passive.x_d1'), axis=-1)
    assert np.all(speeds[-1] == pytest.approx(speeds[0] * np.exp(-0.05 * 10.0), rel=0.01))


def test_active_passive_flocking_brings_active_agents_to_their_speed():
    """Test that an active agent without contact, noise, or flocking relaxes to its preferred speed."""
    system = active_passive_flocking(3, active_share=1.0, alpha=0.0, noise=0.0, packing_fraction=0.001, mu=1.0)
    trajectory = Simulation(system, dt=1.0, random_state=0).check().prepare().run(until=20.0).trajectory_
    assert np.linalg.norm(trajectory.values('active.x_d1')[-1], axis=-1) == pytest.approx(1.0, rel=1e-3)


def test_active_passive_flocking_repels_overlapping_agents():
    """Test that two overlapping passive agents separate under the Hertzian force."""
    system = active_passive_flocking(3, active_share=1 / 3, alpha=0.0, noise=0.0, packing_fraction=0.01)
    positions = np.array([[5.0, 5.0], [6.0, 5.0]])
    trajectory = (
        Simulation(
            system,
            dt=0.1,
            initial_state={'passive.x': positions, 'passive.x_d1': np.zeros((2, 2))},
        )
        .check()
        .prepare()
        .run(until=0.1)
        .trajectory_
    )
    final = trajectory.values('passive.x')[-1]
    assert final[1, 0] - final[0, 0] > 1.0


def test_active_passive_flocking_measures_distance_in_the_periodic_box():
    """Test that the observables record distances within half of the diagonal of the box."""
    trajectory = run('active_passive_flocking', n_agents=40, settings={'until': 2.0}).trajectory_
    size = active_passive_flocking(40).space.extent[0]
    assert np.all(trajectory.values('active_distance') < size / np.sqrt(2))
    assert np.all(trajectory.values('passive_distance') > 0.0)


def test_active_passive_flocking_rejects_no_active_agent():
    """Test that a share that leaves no active agent raises an error naming it."""
    with pytest.raises(ValidationError, match='no active agent'):
        active_passive_flocking(10, active_share=0.0)
