"""Test the `_potts_sampling` module of `skcomplex.systems.physics.common.spins`."""

import numpy as np

from skcomplex.systems.physics.common.spins import PottsSamplingInteraction

from .conftest import _run


def test_potts_spins_take_only_their_states_and_order_cold():
    """Test that Potts spins stay within their `q` states and order into one state at a low temperature."""
    colors = np.random.default_rng(1).integers(0, 3, (12, 12)).astype(float)
    potts = PottsSamplingInteraction('potts', target='sites', q=3)
    values = _run(potts, 'color', {'color': colors, 'temperature': 0.1})
    assert set(np.unique(values)) <= {0.0, 1.0, 2.0}
    assert np.bincount(values[-1].astype(int).ravel()).max() > 0.6 * values[-1].size
