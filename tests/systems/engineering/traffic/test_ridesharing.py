"""Test the `_ridesharing` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.engineering.traffic import RidesharingSystem


def test_ridesharing_waits_grow_near_saturation():
    """Test that riders wait far longer and every cab is busy once the demand exceeds what the fleet serves."""
    calm = simulate(RidesharingSystem(demand=0.3), until=150, random_state=0)
    rush = simulate(RidesharingSystem(demand=1.2), until=150, random_state=0)
    assert np.nanmean(rush.get_values('mean_wait')[100:]) > 5 * np.nanmean(calm.get_values('mean_wait')[100:])
    assert rush.get_values('utilisation')[100:].mean() > 0.9 > calm.get_values('utilisation')[100:].mean()
    assert rush.get_values('queue')[-1] > calm.get_values('queue')[-1]


def test_ranking_values_of_another_kind_raise_a_validation_error():
    """Test that the ranks of riders and cabs reject values that are not riders and cabs."""
    system = RidesharingSystem()
    with pytest.raises(ValidationError, match='A rider ranks cabs'):
        system._get_nearness(1, 2)
    with pytest.raises(ValidationError, match='A cab ranks riders'):
        system._get_patience(1, 2)
