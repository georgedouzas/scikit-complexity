"""Test the `_queues` module of `skcomplex.models.mathematics.stochastic_processes`."""

import numpy as np

from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.stochastic_processes import birth_death_queue


def test_birth_death_queue_stationary_distribution():
    """Test that an M/M/1 queue of load 1/2 occupies n customers a fraction (1 - rho) rho^n of the time."""
    rho = 0.5
    system = birth_death_queue(200, arrival_rate=rho, service_rate=1.0)
    simulation = Simulation(system, dt=0.5, random_state=0, backend=NumbaBackend()).run(until=2000)
    lengths = simulation.trajectory_.values('queues.n')[100:]
    occupation = np.array([np.mean(lengths == n) for n in range(6)])
    assert np.all(np.abs(occupation - (1 - rho) * rho ** np.arange(6)) < 0.01)
