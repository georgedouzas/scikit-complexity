"""Test the `_trails` module of `skcomplex.systems.biology.collective_behaviour`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import (
    ActiveWalkersSystem,
)


def test_active_walkers_bundle_trails():
    """Test that attraction to trails makes walkers wear fewer cells of the green than walkers who ignore trails."""
    worn = []
    for attraction in (0.0, 1.0):
        trajectory = simulate(
            ActiveWalkersSystem(attraction=attraction).prepare().system_,
            until=300,
            dt=1,
            random_state=0,
        )
        worn.append(int((trajectory.get_values('land/ground.G')[-1] > 0.5).sum()))
    assert worn[1] < 0.8 * worn[0]
