"""Test the `_conversations` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np

from skcomplex.systems.social.opinion_dynamics import (
    settle,
)


def test_settling_lowers_the_energy():
    """Test that settling the belief network does not raise its energy and keeps valences in range."""
    coherence = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])
    start = np.array([0.0, 0.5, 0.5])
    settled = settle(start)
    assert -settled @ coherence @ settled <= -start @ coherence @ start
    assert np.abs(settled).max() <= 1
