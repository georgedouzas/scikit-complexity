"""Test the `_mobility` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.organizations import container_mobility, exploration_preferential_return

from ..conftest import _run, _values


def test_exploration_grows_the_distinct_locations_sublinearly():
    """Test that the number of distinct locations grows as `n^(1 / (1 + gamma))` with the displacements."""
    trajectory = _run(exploration_preferential_return(50), 500)
    early, late = (np.mean(_values(trajectory, 'individuals', 'distinct', frame)) for frame in (50, 500))
    assert np.log(late / early) / np.log(10) == pytest.approx(1 / 1.21, abs=0.08)


def test_container_mobility_travels_at_the_level_distances_with_their_probabilities():
    """Test that displacements happen at every level with its probability and grow with the level."""
    trajectory = _run(container_mobility(20), 300)
    levels = np.array([_values(trajectory, 'individuals', 'level', frame) for frame in range(1, 301)])
    np.testing.assert_allclose([np.mean(levels == level) for level in (1, 2, 3)], [0.6, 0.3, 0.1], atol=0.02)
    x = np.array([_values(trajectory, 'individuals', 'x', frame) for frame in range(1, 301)])
    y = np.array([_values(trajectory, 'individuals', 'y', frame) for frame in range(1, 301)])
    lengths, kept = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)), levels[1:]
    means = [lengths[kept == level].mean() for level in (1, 2, 3)]
    assert means[0] < means[1] < means[2]
    assert np.all(lengths[kept == 1] < 3 * np.sqrt(2) + 1e-9)


def test_container_mobility_rejects_travel_that_does_not_match_the_levels():
    """Test that the probabilities of travel must give one probability per level."""
    with pytest.raises(ValidationError, match='probabilities of travel'):
        _run(container_mobility(2, travel=(0.5, 0.5)), 1)
