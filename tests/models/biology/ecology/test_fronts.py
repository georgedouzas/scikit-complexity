"""Test the `_fronts` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.ecology import fisher_kpp


def test_fisher_front_speed():
    """Test that the Fisher-KPP front moves at a speed within 5% of 2 sqrt(r D)."""
    for r, diffusion in [(1.0, 1.0), (0.5, 2.0)]:
        trajectory = Simulation(fisher_kpp(D=diffusion, r=r), dt=2.0).run(until=80).trajectory_
        positions = np.arange(800) * 0.25
        fronts = np.asarray([positions[np.argmax(values < 0.5)] for values in trajectory.values('line.u')])
        late = trajectory.times >= 40
        speed = np.polyfit(trajectory.times[late], fronts[late], 1)[0]
        assert speed == pytest.approx(2 * np.sqrt(r * diffusion), rel=0.05)
