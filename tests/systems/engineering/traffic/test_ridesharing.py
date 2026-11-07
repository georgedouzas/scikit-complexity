"""Test the `_ridesharing` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.traffic import RidesharingSystem


def test_ridesharing_waits_grow_near_saturation():
    """Test that riders wait far longer and every cab is busy once the demand exceeds what the fleet serves."""
    calm = simulate(RidesharingSystem(demand=0.3), until=150)
    rush = simulate(RidesharingSystem(demand=1.2), until=150)
    assert np.nanmean(rush.values('mean_wait')[100:]) > 5 * np.nanmean(calm.values('mean_wait')[100:])
    assert rush.values('utilisation')[100:].mean() > 0.9 > calm.values('utilisation')[100:].mean()
    assert rush.values('queue')[-1] > calm.values('queue')[-1]
