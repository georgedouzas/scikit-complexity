"""Test the `_cell_quota` module of `skcomplex.models.biology.ecology`."""

import numpy as np

from skcomplex.models.biology.ecology import droop_cell_quota

from ..conftest import _run_seeded


def test_droop_cell_quota_settles_and_grows_exponentially():
    """Test that the quota settles at `q + v / r_inf` and the biomass grows at `r_inf (1 - q / Q*) - d`."""
    trajectory = _run_seeded(droop_cell_quota(), until=40.0, dt=0.5)
    quota = 1.0 + 2.0 * 1.0 / (1.0 + 1.0) / 1.0
    assert abs(trajectory['algae.Q'][-1] - quota) < 1e-6
    late = trajectory.times >= 30.0
    slope = np.polyfit(trajectory.times[late], np.log(trajectory['algae.P'][late]), 1)[0]
    assert abs(slope - (1.0 * (1 - 1.0 / quota) - 0.1)) < 1e-6
