"""Test the `_bounded_confidence` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import LatticeElement
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.social.opinion_dynamics import (
    ConfidenceCouplingInteraction,
    ConfidentElement,
    DeffuantSystem,
    HegselmannKrauseSystem,
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
    system = HegselmannKrauseSystem(300, confidence=confidence).prepare().system_
    opinions = Simulation(system, random_state=0).prepare().run(until=40).trajectory_.values('people.opinion')
    found = np.unique(np.round(opinions[-1], 6))
    assert abs(found.size - clusters) <= 1
    assert np.allclose(opinions[-1], opinions[-2], atol=1e-12)
    assert np.all(np.diff(found) > confidence)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_deffuant_bounded_confidence_clusters(seed):
    """Test that Deffuant opinions with confidence 0.2 converge to about `1 / (2 * 0.2)` clusters."""
    trajectory = _run(DeffuantSystem(500, confidence=0.2, random_state=seed), 100)
    opinions = _values(trajectory, 'people', 'opinion')
    sizes = _clusters(opinions, 0.05)
    major = sizes[sizes >= 0.05 * len(opinions)]
    assert len(major) in {2, 3}
    assert 2 <= len(sizes) <= 4
    initial = _clusters(_values(trajectory, 'people', 'opinion', 0), 0.05)
    assert len(initial) == 1


def test_deffuant_conserves_mean_opinion():
    """Test that compromises keep the mean opinion of the Deffuant system."""
    trajectory = _run(DeffuantSystem(100, random_state=0), 20)
    first, last = _values(trajectory, 'people', 'opinion', 0), _values(trajectory, 'people', 'opinion')
    assert last.mean() == pytest.approx(first.mean())


def test_confidence_coupling_takes_the_mean_of_the_trusted_neighbours():
    """Test that one step gives every agent the mean opinion of the trusted neighbours, itself included."""
    opinions = np.array([0.1, 0.2, 0.35, 0.9, 0.95])
    people = LatticeElement(
        'people',
        element=ConfidentElement,
        space=NetworkSpace(nx.complete_graph(5)),
        values={'opinion': opinions, 'confidence': 0.2},
    )
    system = System(
        elements=[people],
        interactions=[ConfidenceCouplingInteraction('compromise', target='people')],
        time=DiscreteTime(),
    )
    values = Simulation(system).prepare().run(until=1).trajectory_.values('people.opinion')[-1]
    close = np.abs(opinions[None, :] - opinions[:, None]) < 0.2
    np.testing.assert_allclose(values, (close * opinions).sum(1) / close.sum(1), atol=1e-12)
