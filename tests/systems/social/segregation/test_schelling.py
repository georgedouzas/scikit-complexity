"""Test the `_schelling` module of `skcomplex.systems.social.segregation`."""

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.social.segregation import schelling


def _similar_share(snapshot):
    """Return the mean share of a resident's neighbours that have its kind."""
    kinds = {
        tuple(location): kind
        for location, kind in zip(snapshot.locations.tolist(), snapshot.values['kind'], strict=True)
    }
    shares = []
    for (column, row), kind in kinds.items():
        neighbors = [
            kinds[(column + dx, row + dy)]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy) and (column + dx, row + dy) in kinds
        ]
        if neighbors:
            shares.append(np.mean([neighbor == kind for neighbor in neighbors]))
    return float(np.mean(shares))


def test_schelling_segregation_increases():
    """Test that the share of similar neighbours is higher after 100 steps than at the start."""
    trajectory = (
        Simulation(schelling(20, 20, threshold=0.5, random_state=0), random_state=0)
        .prepare()
        .run(until=100)
        .trajectory_
    )
    assert _similar_share(trajectory.agents('residents', 100)) > _similar_share(trajectory.agents('residents', 0))


def test_big_sort_segregation_grows_with_the_number_of_choices():
    """Test that residents who pick the best of a few empty cells segregate, more with more choices, and not with one."""
    shares = {}
    for choices in (1, 4, 10):
        system = schelling(20, 20, choices=choices, random_state=0)
        trajectory = Simulation(system, random_state=0).prepare().run(until=100).trajectory_
        shares[choices] = _similar_share(trajectory.agents('residents', 100))
    start = _similar_share(Simulation(schelling(20, 20), random_state=0).prepare().run(until=0).trajectory_.agents('residents', 0))
    assert abs(shares[1] - start) < 0.05
    assert start < shares[4] < shares[10]
