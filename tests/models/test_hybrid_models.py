"""Test the delay, event, and noise models of the library."""

import math

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.neuroscience import integrate_and_fire, izhikevich
from skcomplex.models.physics.mechanics import circular_billiard, newtons_cradle
from skcomplex.models.physics.soft_matter import brownian_near_wall, stochastic_resonance
from skcomplex.models.physics.synchronization import delayed_kuramoto


def test_izhikevich_regular_spiking():
    """Test that the regular-spiking Izhikevich neuron with input 10 fires periodically and records every reset."""
    trajectory = Simulation(izhikevich(a=0.02, b=0.2, c=-65, d=8, I=10), dt=0.1).run(until=400).trajectory_
    spikes = [event for event in trajectory.events if event.label == 'spike']
    intervals = np.diff([event.time for event in spikes])
    assert len(spikes) > 5
    assert np.ptp(intervals[2:]) < 0.1
    assert all(event.after[0] == -65 and event.after[1] == pytest.approx(event.before[1] + 8) for event in spikes)


def test_integrate_and_fire_period():
    """Test that the leaky integrate-and-fire neuron spikes with the analytic period."""
    trajectory = Simulation(integrate_and_fire(), dt=0.1).run(until=100).trajectory_
    intervals = np.diff([0.0] + [event.time for event in trajectory.events])
    assert intervals == pytest.approx(10 * math.log(20 / 5), abs=1e-5)


def test_delayed_kuramoto_synchronizes():
    """Test that strongly coupled delayed oscillators lock their phases."""
    trajectory = Simulation(delayed_kuramoto(K=2.0, tau=1.0), dt=0.1).run(until=50).trajectory_
    order = np.abs(np.exp(1j * trajectory.states).mean(axis=1))
    assert order[0] < 0.1
    assert order[-1] > 0.99


def test_newtons_cradle_passes_momentum():
    """Test that the cradle conserves momentum through collisions and moves the last ball."""
    trajectory = Simulation(newtons_cradle(n_balls=3), dt=0.01).run(until=0.2).trajectory_
    velocities = trajectory.states[:, 1::2]
    assert [event.label for event in trajectory.events][:2] == ['collision_0', 'collision_1']
    assert velocities[-1, -1] > 0.4
    assert velocities[-1, :-1] == pytest.approx([0.0, 0.0], abs=0.02)


def test_circular_billiard_keeps_speed_and_angle():
    """Test that the billiard keeps its speed and its angle of incidence at every bounce."""
    trajectory = Simulation(circular_billiard(), dt=0.05).run(until=20).trajectory_
    speeds = np.hypot(trajectory['particle.vx'], trajectory['particle.vy'])
    assert speeds == pytest.approx(np.full(speeds.size, math.hypot(0.4, 1.0)))
    assert np.hypot(trajectory['particle.x'], trajectory['particle.y']).max() <= 1.0 + 1e-9
    incidence = [
        abs(event.before[0] * event.before[2] + event.before[1] * event.before[3]) for event in trajectory.events
    ]
    assert len(incidence) > 5
    assert np.ptp(incidence) < 1e-6


def test_stochastic_resonance_hops_between_wells():
    """Test that noise moves the particle between the wells of the double well."""
    values = Simulation(stochastic_resonance(), dt=0.5, random_state=0).run(until=300).trajectory_['particle.x']
    assert values.min() < -0.5
    assert values.max() > 0.5


def test_brownian_particle_stays_off_the_wall():
    """Test that the particle near the reflecting wall keeps a positive distance and finite values."""
    values = Simulation(brownian_near_wall(), dt=0.1, random_state=0).run(until=10).trajectory_['particle.z']
    assert np.all(np.isfinite(values))
    assert values.min() >= 0.0
