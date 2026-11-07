"""Test the `_cell_quota` module of `skcomplex.systems.biology.ecology`."""

import numpy as np

from skcomplex.systems.biology.ecology import DroopCellQuotaSystem

from ..conftest import _run_seeded


def test_droop_cell_quota_settles_and_grows_exponentially():
    """Test that the quota settles at `q + v / r_inf` and the biomass grows at `r_inf (1 - q / Q*) - d`."""
    trajectory = _run_seeded(DroopCellQuotaSystem(), until=40.0, dt=0.5)
    quota = 1.0 + 2.0 * 1.0 / (1.0 + 1.0) / 1.0
    assert abs(trajectory.values('algae.Q')[-1] - quota) < 1e-6
    late = trajectory.times >= 30.0
    slope = np.polyfit(trajectory.times[late], np.log(trajectory.values('algae.P')[late]), 1)[0]
    assert abs(slope - (1.0 * (1 - 1.0 / quota) - 0.1)) < 1e-6
