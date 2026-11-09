"""Test the `_records` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import RecordDynamicsEpidemicSystem


def test_outbreaks_follow_records_of_the_strain_strength():
    """Test that the mean number of outbreaks in 100 years is the harmonic number of the 99 strains drawn."""
    outbreaks = []
    for seed in range(30):
        trajectory = simulate(
            RecordDynamicsEpidemicSystem(2000, seed=seed).prepare().system_,
            until=100,
            random_state=seed,
        )
        infected = trajectory.get_values('residents.infected').sum(axis=1)
        outbreaks.append((infected > 0).sum())
    assert abs(np.mean(outbreaks) - sum(1 / k for k in range(1, 100))) < 0.6


def test_infected_residents_become_immune():
    """Test that every infected resident has the resistance 1 and is never infected again."""
    trajectory = simulate(RecordDynamicsEpidemicSystem(500), random_state=3, until=200)
    infected, resistance = trajectory.get_values('residents.infected'), trajectory.get_values('residents.R')
    assert np.all(resistance[infected == 1] == 1.0)
    assert np.all(infected.sum(axis=0) <= 1)
