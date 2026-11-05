"""Test the `_trails` module of `skcomplex.models.biology.collective_behaviour`."""

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import (
    active_walkers,
)


def test_active_walkers_bundle_trails():
    """Test that attraction to trails makes walkers wear fewer cells of the green than walkers who ignore trails."""
    worn = []
    for attraction in (0.0, 1.0):
        simulation = Simulation(active_walkers(attraction=attraction), dt=1, random_state=0).prepare()
        trajectory = simulation.run(until=300).trajectory_
        worn.append(int((trajectory.values('land/ground.G')[-1] > 0.5).sum()))
    assert worn[1] < 0.8 * worn[0]
