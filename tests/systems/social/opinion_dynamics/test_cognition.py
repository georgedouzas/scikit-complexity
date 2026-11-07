"""Test the `_cognition` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np
import pytest

from skcomplex.systems.social.opinion_dynamics import (
    CognitiveConsistencySystem,
    settle,
)

from ..conftest import _run, _values


@pytest.mark.parametrize(('positive', 'sign'), [(0.8, 1.0), (0.2, -1.0)])
def test_cognitive_consistency_agrees_with_the_majority(positive, sign):
    """Test that a complete graph agrees on an attitude whose sign is that of the initial majority."""
    trajectory = _run(CognitiveConsistencySystem(positive=positive, random_state=0), 60)
    valences = _values(trajectory, 'thinkers', 'valences')
    assert np.sign(valences[:, 1]).tolist() == [sign] * len(valences)
    assert valences.std(axis=0).max() < 0.05
    assert np.abs(valences).max() <= 1


def test_settling_lowers_the_energy():
    """Test that settling the belief network does not raise its energy and keeps valences in range."""
    coherence = np.array([[0.0, 0.05, 0.0], [0.05, 0.0, -0.05], [0.0, -0.05, 0.0]])
    start = np.array([0.0, 0.5, 0.5])
    settled = settle(start)
    assert -settled @ coherence @ settled <= -start @ coherence @ start
    assert np.abs(settled).max() <= 1
