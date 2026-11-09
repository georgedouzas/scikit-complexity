"""Test the `_active_passive` module of `skcomplex.systems.physics.active_matter`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.active_matter import ActivePassiveFlockingSystem

LAST_TEN_TIME_UNITS = 20


def _distances(alpha, seed):
    """Return the mean distances within the active and the passive agents at the end of a run of figure 10.7."""
    settings = {'until': 50.0, 'dt': 0.5, 'random_state': seed}
    found = simulate(ActivePassiveFlockingSystem(alpha=alpha, random_state=seed), **settings)
    return tuple(
        found.get_values(name)[-LAST_TEN_TIME_UNITS:].mean() for name in ('active_distance', 'passive_distance')
    )


def test_active_passive_flocking_flocks_when_every_agent_is_active():
    """Test that the polarization of the velocities rises above 0.9 with every agent active and flocking on."""
    settings = {'until': 30.0, 'dt': 0.5, 'random_state': 1}
    found = simulate(ActivePassiveFlockingSystem(n_agents=114, n_active=114), **settings)
    polarization = found.get_values('polarization')
    assert polarization[0] < 0.3
    assert polarization[-LAST_TEN_TIME_UNITS:].mean() > 0.9


def test_active_passive_flocking_gathers_active_agents_with_flocking_on():
    """Test that the settings of figure 10.7 bring the active agents closer together, as a mean over seeds."""
    active, passive = np.mean([_distances(1.0, seed) for seed in range(4)], axis=0)
    assert active < passive - 0.015


def test_active_passive_flocking_does_not_gather_active_agents_with_flocking_off():
    """Test that without the flocking force the two mean distances stay within 0.01, as a mean over seeds."""
    active, passive = np.mean([_distances(0.0, seed) for seed in range(4)], axis=0)
    assert active == pytest.approx(passive, abs=0.01)


def test_active_passive_flocking_keeps_passive_agents_near_rest():
    """Test that passive agents move far slower than active ones, as the polar diagrams of figure 10.7 show."""
    settings = {'until': 20.0, 'dt': 0.5, 'random_state': 0}
    velocities = simulate(ActivePassiveFlockingSystem(n_agents=114), **settings).get_values('agents.x_d1')[-1]
    speeds = np.linalg.norm(velocities, axis=-1)
    assert speeds[:45].mean() > 3 * speeds[45:].mean()


def test_active_passive_flocking_slows_passive_agents():
    """Test that passive agents without contact lose their speed at the rate of the propulsion."""
    system = ActivePassiveFlockingSystem(4, n_active=1, alpha=0.0, eta=0.0, size=100.0, initial_speed=0.1)
    trajectory = simulate(system, until=0.2, dt=0.05, solver=EulerMaruyamaSolver(), random_state=0)
    speeds = np.linalg.norm(trajectory.get_values('agents.x_d1')[:, 1:], axis=-1)
    assert speeds[-1] == pytest.approx(speeds[0] * np.exp(-10.0 * 0.2), rel=0.02)


def test_active_passive_flocking_brings_active_agents_to_their_speed():
    """Test that an active agent without contact, noise, or flocking relaxes to its target speed."""
    system = ActivePassiveFlockingSystem(3, n_active=3, alpha=0.0, eta=0.0, size=100.0)
    trajectory = simulate(system, until=2.0, dt=0.1, random_state=0)
    assert np.linalg.norm(trajectory.get_values('agents.x_d1')[-1], axis=-1) == pytest.approx(0.02, rel=1e-3)


def test_active_passive_flocking_flocks_from_the_velocities_of_both_kinds_without_itself():
    """Test that an active agent follows the velocity of a passive neighbour, and feels nothing with no neighbour."""
    system = ActivePassiveFlockingSystem(2, n_active=1, v0=0.0, mu=0.0, eta=0.0, epsilon=0.0, size=1.0)
    start = {'agents.x': np.array([[0.5, 0.5], [0.55, 0.5]]), 'agents.x_d1': np.array([[0.0, 0.0], [0.0, 1.0]])}
    near = simulate(system, until=0.01, dt=0.01, initial_state=start)
    assert near.get_values('agents.x_d1')[-1][0].tolist() == pytest.approx([0.0, 0.01], abs=1e-6)
    alone = {'agents.x': np.array([[0.1, 0.1], [0.6, 0.6]]), 'agents.x_d1': np.array([[1.0, 0.0], [0.0, 1.0]])}
    apart = simulate(system, until=0.01, dt=0.01, initial_state=alone)
    assert apart.get_values('agents.x_d1')[-1][0].tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_active_passive_flocking_repels_with_the_force_of_the_book():
    """Test that the repulsion has the size `epsilon (1 - r / (2 r_0))^(3/2)` and pushes overlapping agents apart."""
    system = ActivePassiveFlockingSystem(2, n_active=1, alpha=0.0, eta=0.0, mu=0.0)
    gap = 0.025
    start = {'agents.x': np.array([[0.5, 0.5], [0.5 + gap, 0.5]]), 'agents.x_d1': np.zeros((2, 2))}
    final = simulate(system, until=1e-4, dt=1e-4, initial_state=start)
    force = 25.0 * (1 - gap / 0.05) ** 1.5
    assert final.get_values('agents.x_d1')[-1][1, 0] == pytest.approx(1e-4 * force, rel=1e-3)
    assert final.get_values('agents.x_d1')[-1][0, 0] == pytest.approx(-1e-4 * force, rel=1e-3)


def test_active_passive_flocking_marks_the_active_agents_first():
    """Test that the group holds the given number of active agents first, the rest passive."""
    system = ActivePassiveFlockingSystem(10, n_active=4).prepare().system_
    assert system.elements[0].values['active'].tolist() == [1.0] * 4 + [0.0] * 6


def test_active_passive_flocking_takes_the_noise_per_step_of_the_book():
    """Test that the noise on a free agent has the variance per step of a uniform force of half-width `eta`."""
    system = ActivePassiveFlockingSystem(
        200,
        n_active=200,
        alpha=0.0,
        v0=0.0,
        mu=0.0,
        epsilon=0.0,
        eta=1.0,
        initial_speed=0.0,
    )
    trajectory = simulate(system, until=1.0, dt=1.0, random_state=0)
    variance = trajectory.get_values('agents.x_d1')[-1].var()
    assert variance == pytest.approx(100 * 0.01**2 / 3, rel=0.2)


def test_active_passive_flocking_measures_distance_in_the_periodic_box():
    """Test that the observables record distances within half of the diagonal of the box."""
    trajectory = simulate(ActivePassiveFlockingSystem(n_agents=40, n_active=20), until=2.0, random_state=0)
    assert np.all(trajectory.get_values('active_distance') < 1.0 / np.sqrt(2))
    assert np.all(trajectory.get_values('passive_distance') > 0.0)


def test_active_passive_flocking_rejects_a_wrong_number_of_active_agents():
    """Test that a number of active agents outside the range raises an error naming it."""
    with pytest.raises(ValidationError, match='0 active agents'):
        ActivePassiveFlockingSystem(10, n_active=0).prepare()
    with pytest.raises(ValidationError, match='11 active agents'):
        ActivePassiveFlockingSystem(10, n_active=11).prepare()


def test_active_passive_flocking_reads_only_active_noise():
    """Test that the amplitude of the random force of a passive agent is 0."""
    noise = ActivePassiveFlockingSystem(4, n_active=2).prepare().system_.interactions[1].noise
    assert noise(SimpleNamespace(target=SimpleNamespace(active=0.0))) == 0.0
