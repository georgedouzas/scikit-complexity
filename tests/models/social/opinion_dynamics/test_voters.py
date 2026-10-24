"""Test the `_voters` module of `skcomplex.models.social.opinion_dynamics`."""

import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models.social.opinion_dynamics import (
    majority_rule,
    voter_model,
    voter_variants,
)

from ..conftest import _run, _values


def test_voter_model_reaches_consensus():
    """Test that 20 voters on a connected network end with one opinion."""
    graph = nx.connected_watts_strogatz_graph(20, 4, 0.2, seed=0)
    trajectory = Simulation(voter_model(graph, random_state=0), random_state=0).run(until=500).trajectory_
    assert len(set(trajectory.agents('voters', 500).values['opinion'].tolist())) == 1


def test_majority_rule_settles():
    """Test that synchronous majority voting settles into domains that are fixed or blink with period two."""
    trajectory = Simulation(majority_rule((50, 50)), dt=1).run(until=50).trajectory_
    opinions = trajectory['voters.opinion']
    changes = np.abs(np.diff(opinions, axis=0)).sum(axis=1)
    assert changes[0] > 100
    assert changes[-1] < 25
    assert np.array_equal(opinions[-1], opinions[-3])


def test_voter_variants_contrarians_prevent_consensus():
    """Test that majority voters reach consensus, and contrarians keep a mixed state."""
    graph = nx.complete_graph(30)
    majority = _run(voter_variants(graph, contrarians=0.0, random_state=0), 50)
    assert len(set(_values(majority, 'voters', 'opinion').tolist())) == 1
    mixed = _run(voter_variants(graph, contrarians=0.2, random_state=0), 50)
    assert len(set(_values(mixed, 'voters', 'opinion').tolist())) == 2
