"""Test the `_bounded_confidence` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import LatticeElement
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
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
    opinions = simulate(system, until=40, random_state=0).values('people.opinion')
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
    values = simulate(system, until=1).values('people.opinion')[-1]
    close = np.abs(opinions[None, :] - opinions[:, None]) < 0.2
    np.testing.assert_allclose(values, (close * opinions).sum(1) / close.sum(1), atol=1e-12)


def test_deffuant_large_steepness_matches_the_threshold():
    """Test that steepness 1e4 gives the opinions of the deterministic threshold from the same seed."""
    threshold = _run(DeffuantSystem(confidence=0.2), 50)
    steep = _run(DeffuantSystem(confidence=0.2, steepness=1e4), 50)
    np.testing.assert_allclose(_values(steep, 'people', 'opinion'), _values(threshold, 'people', 'opinion'), atol=1e-6)


def test_deffuant_records_one_outcome_per_encounter_with_its_probability():
    """Test that a recorded run holds one Bernoulli outcome per encounter with its compromise probability."""
    trajectory = simulate(DeffuantSystem(50, steepness=20.0), until=10, random_state=0, record_draws=True)
    record = trajectory.draws.draws['meetings.outcome']
    assert record.kind == 'bernoulli'
    assert record.values.size == 50 * 10
    assert set(np.unique(record.values)) <= {0.0, 1.0}
    assert np.all((record.arguments['p'] > 0) & (record.arguments['p'] < 1))
    assert np.bincount(record.steps).tolist() == [0] + [50] * 10


def test_deffuant_replay_keeps_the_opinions():
    """Test that replaying a recorded run with another confidence keeps the recorded opinions."""
    recorded = simulate(DeffuantSystem(50, steepness=20.0), until=10, random_state=0, record_draws=True)
    replayed = simulate(DeffuantSystem(50, confidence=0.3, steepness=20.0), until=10, random_state=0, replay=recorded)
    np.testing.assert_array_equal(_values(replayed, 'people', 'opinion'), _values(recorded, 'people', 'opinion'))
    assert replayed.log_probability != pytest.approx(recorded.log_probability)
