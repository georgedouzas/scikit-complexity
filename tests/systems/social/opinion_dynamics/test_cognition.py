"""Test the `_cognition` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np
import pytest

from skcomplex.systems.social.opinion_dynamics import (
    cognitive_consistency,
)

from ..conftest import _run, _values


@pytest.mark.parametrize(('positive', 'sign'), [(0.8, 1.0), (0.2, -1.0)])
def test_cognitive_consistency_agrees_with_the_majority(positive, sign):
    """Test that a complete graph agrees on an attitude whose sign is that of the initial majority."""
    trajectory = _run(cognitive_consistency(positive=positive, random_state=0), 60)
    valences = _values(trajectory, 'thinkers', 'valences')
    assert np.sign(valences[:, 1]).tolist() == [sign] * len(valences)
    assert valences.std(axis=0).max() < 0.05
    assert np.abs(valences).max() <= 1
