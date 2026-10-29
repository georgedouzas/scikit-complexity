"""Test the `_leadership` module of `skcomplex.models.biology.collective_behaviour`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.collective_behaviour import couzin_leadership


def _accuracy(informed):
    """Return the mean cosine between the shoal's late heading and the preferred direction, over four shoals."""
    cosines = []
    for seed in range(4):
        trajectory = (
            Simulation(
                couzin_leadership(30, informed=informed, random_state=seed),
                dt=1,
                random_state=seed,
            )
            .check()
            .prepare()
        )
        trajectory = trajectory.run(150).trajectory_
        for frame in range(100, 151):
            snapshot = trajectory.agents('shoal', frame)
            heading = np.arctan2(np.mean(snapshot.values['vy']), np.mean(snapshot.values['vx']))
            cosines.append(np.cos(heading))
    return float(np.mean(cosines))


def test_couzin_leadership_informed_minority_leads():
    """Test that an informed minority turns the shoal toward its direction, more accurately when it is larger."""
    naive, few, many = _accuracy(0.0), _accuracy(0.1), _accuracy(0.3)
    assert naive < 0.5
    assert 0.6 < few < many
