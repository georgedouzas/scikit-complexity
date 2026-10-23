"""Test the `_ridesharing` module of `skcomplex.models.engineering.traffic`."""

import numpy as np

from skcomplex.models import run


def test_ridesharing_waits_grow_near_saturation():
    """Test that riders wait far longer and every cab is busy once the demand exceeds what the fleet serves."""
    calm = run('ridesharing', demand=0.3, settings={'until': 150}).trajectory_
    rush = run('ridesharing', demand=1.2, settings={'until': 150}).trajectory_
    assert np.nanmean(rush['mean_wait'][100:]) > 5 * np.nanmean(calm['mean_wait'][100:])
    assert rush['utilisation'][100:].mean() > 0.9 > calm['utilisation'][100:].mean()
    assert rush['queue'][-1] > calm['queue'][-1]
