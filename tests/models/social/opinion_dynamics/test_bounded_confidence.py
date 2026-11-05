"""Test the `_bounded_confidence` module of `skcomplex.models.social.opinion_dynamics`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.social.opinion_dynamics import (
    deffuant,
    hegselmann_krause,
)

from ..conftest import _run, _values


def _clusters(opinions, gap):
    """Return the sizes of the groups of sorted opinions separated by more than a gap."""
    ordered = np.sort(opinions)
    labels = np.concatenate([[0], np.cumsum(np.diff(ordered) > gap)])
    return np.bincount(labels)


@pytest.mark.parametrize(('confidence', 'clusters'), [(0.3, 1), (0.15, 3), (0.08, 6)])
def test_hegselmann_krause_freezes_into_clusters_of_the_confidence(confidence, clusters):
    """Test that the opinions settle into about `1 / (2 ε)` clusters that stop moving."""
    system = hegselmann_krause(300, confidence=confidence)
    opinions = Simulation(system, random_state=0).prepare().run(until=40).trajectory_.values('people.opinion')
    found = np.unique(np.round(opinions[-1], 6))
    assert abs(found.size - clusters) <= 1
    assert np.allclose(opinions[-1], opinions[-2], atol=1e-12)
    assert np.all(np.diff(found) > confidence)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_deffuant_bounded_confidence_clusters(seed):
    """Test that Deffuant opinions with confidence 0.2 converge to about `1 / (2 * 0.2)` clusters."""
    trajectory = _run(deffuant(500, confidence=0.2, random_state=seed), 100)
    opinions = _values(trajectory, 'people', 'opinion')
    sizes = _clusters(opinions, 0.05)
    major = sizes[sizes >= 0.05 * len(opinions)]
    assert len(major) in {2, 3}
    assert 2 <= len(sizes) <= 4
    initial = _clusters(_values(trajectory, 'people', 'opinion', 0), 0.05)
    assert len(initial) == 1


def test_deffuant_conserves_mean_opinion():
    """Test that compromises keep the mean opinion of the Deffuant model."""
    trajectory = _run(deffuant(100, random_state=0), 20)
    first, last = _values(trajectory, 'people', 'opinion', 0), _values(trajectory, 'people', 'opinion')
    assert last.mean() == pytest.approx(first.mean())
