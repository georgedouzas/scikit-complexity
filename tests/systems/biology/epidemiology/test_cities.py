"""Test the `_cities` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import EpisimsCitySystem


def _final_size(closed: bool, seed: int) -> float:
    """Return the share of recovered people after 200 half days."""
    trajectory = simulate(
        EpisimsCitySystem(500, closed=closed, seed=seed).prepare().system_,
        until=200,
        random_state=seed,
    )
    return float((trajectory.agents('people', 200).values['health'] == 2).mean())


def test_workplaces_spread_the_disease_between_households():
    """Test that the epidemic reaches most of the city, and that closing workplaces confines it to households."""
    open_sizes = [_final_size(False, seed) for seed in range(2)]
    closed_sizes = [_final_size(True, seed) for seed in range(2)]
    assert min(open_sizes) > 0.8
    assert max(closed_sizes) < 0.1


def test_people_alternate_between_home_and_work():
    """Test that people are at home in even half days and at the workplace in odd ones."""
    trajectory = simulate(EpisimsCitySystem(50), random_state=0, until=4)
    agents = trajectory.agents('people', 3)
    np.testing.assert_array_equal(np.asarray(agents.locations).ravel(), agents.values['work'])
    agents = trajectory.agents('people', 4)
    np.testing.assert_array_equal(np.asarray(agents.locations).ravel(), agents.values['home'])
