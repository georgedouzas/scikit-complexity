"""Test the `_herding` module of `skcomplex.systems.biology.collective_behaviour`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.collective_behaviour import ShepherdingSystem


def test_shepherd_brings_the_flock_to_the_goal():
    """Test Strömbom et al.: collecting and driving bring the centre of the flock to the goal within 200 steps."""
    for seed in range(3):
        trajectory = simulate(ShepherdingSystem(seed=seed), random_state=seed, until=200)
        distances = [
            np.linalg.norm(np.asarray(trajectory.agents('sheep', frame).locations, dtype=float).mean(axis=0) - 5.0)
            for frame in range(201)
        ]
        assert distances[0] > 60
        assert min(distances) < 5
        driving = [float(trajectory.agents('shepherds', frame).values['driving'][0]) for frame in range(1, 201)]
        assert np.mean(driving) > 0.5
