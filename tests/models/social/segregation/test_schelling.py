"""Test the `_schelling` module of `skcomplex.models.social.segregation`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.segregation import schelling


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
        .check()
        .run(until=100)
        .trajectory_
    )
    assert _similar_share(trajectory.agents('residents', 100)) > _similar_share(trajectory.agents('residents', 0))
