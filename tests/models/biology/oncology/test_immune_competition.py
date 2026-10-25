"""Test the `_immune_competition` module of `skcomplex.models.biology.oncology`."""

import pytest

from ...conftest import _trajectory


@pytest.mark.parametrize(('activation', 'decays'), [(0.75, True), (1.0, True), (0.0, False), (0.25, False)])
def test_immune_response_above_threshold_clears_the_tumour(activation, decays):
    """Test that the tumour density decays above the immune activation threshold and grows below it."""
    trajectory = _trajectory('immune_competition', 20, 5.0, activation=activation)
    density = trajectory.values('tumour_density')
    assert bool(density[-1] < density[0]) is decays
    assert bool(density[-1] > density[0]) is not decays
