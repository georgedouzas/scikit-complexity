"""Test the `_immune_competition` module of `skcomplex.systems.biology.oncology`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.oncology import ImmuneCompetitionSystem


@pytest.mark.parametrize(('activation', 'decays'), [(0.75, True), (1.0, True), (0.0, False), (0.25, False)])
def test_immune_response_above_threshold_clears_the_tumour(activation, decays):
    """Test that the tumour density decays above the immune activation threshold and grows below it."""
    trajectory = simulate(ImmuneCompetitionSystem(activation=activation), until=20, dt=5.0, random_state=0)
    density = trajectory.get_values('tumour_density')
    assert bool(density[-1] < density[0]) is decays
    assert bool(density[-1] > density[0]) is not decays
