"""Test the `_sequential` module of `skcomplex.systems.biology.ecology`."""

import numpy as np

from ..conftest import _run


def test_sequentially_linear_ecosystem_settles_after_bursts_of_extinctions():
    """Test that abundances stay normalised and extinctions grow rarer as the network settles."""
    trajectory = _run('sequentially_linear_ecosystem', 300, 1.0)
    extinctions = np.nansum(trajectory.values('species.age')[1:] == 0, axis=1)
    assert np.allclose(np.nansum(trajectory.values('species.x')[-1]), 1.0, atol=0.05)
    assert extinctions[:100].sum() > extinctions[200:].sum()
    assert all(trajectory.network(frame).nodes.size == 60 for frame in (0, 150, 300))
