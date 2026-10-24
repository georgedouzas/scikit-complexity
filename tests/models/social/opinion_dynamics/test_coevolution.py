"""Test the `_coevolution` module of `skcomplex.models.social.opinion_dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.engine.analysis import as_graph
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create
from skcomplex.models.social.opinion_dynamics import coevolving_voter, join_a_club


def _discordant(trajectory, frame):
    """Return the share of the edges of a frame whose ends hold different opinions."""
    snapshot = trajectory.network(frame)
    opinions = trajectory['voters.opinion'][frame]
    return float(np.mean(opinions[snapshot.sources] != opinions[snapshot.targets]))


def test_coevolving_voter_fragments_fast_when_rewiring_dominates():
    """Test that fast rewiring removes discordant edges at once and splits opinions, while slow rewiring keeps them."""
    runs = {
        rewiring: Simulation(coevolving_voter(200, rewiring=rewiring), dt=20, random_state=0).run(until=200)
        for rewiring in (0.1, 0.8)
    }
    slow, fast = (runs[rewiring].trajectory_ for rewiring in (0.1, 0.8))
    assert _discordant(slow, 2) > 0.15
    assert _discordant(fast, 2) < 0.02
    graph = as_graph(fast.network(-1))
    opinions = fast['voters.opinion'][-1]
    assert nx.number_connected_components(graph) >= 2
    assert 0 < opinions.mean() < 1
    for component in nx.connected_components(graph):
        assert len({opinions[node] for node in component}) == 1


def test_coevolving_voter_keeps_the_number_of_edges():
    """Test that rewiring moves edges without creating or destroying them."""
    trajectory = Simulation(coevolving_voter(100, rewiring=0.5), dt=10, random_state=1).run(until=50).trajectory_
    sizes = {trajectory.network(frame).sources.size for frame in range(len(trajectory))}
    assert len(sizes) == 1


def test_join_a_club_separates_members_from_outsiders_at_strong_coupling():
    """Test that strong coupling cuts the acquaintances across the club boundary, and zero coupling keeps many."""
    strong = Simulation(join_a_club(200, beta=1.0), dt=10, random_state=0).run(until=150).trajectory_
    free = Simulation(join_a_club(200, beta=0.0), dt=10, random_state=0).run(until=150).trajectory_
    assert _discordant(strong, -1) < 0.02
    assert _discordant(free, -1) > 0.25
    members = strong['voters.opinion'][-1]
    graph = as_graph(strong.network(-1))
    assert nx.number_connected_components(graph) >= 2
    assert 0 < members.mean() < 1


@pytest.mark.parametrize('name', ['hegselmann_krause', 'coevolving_voter', 'join_a_club'])
def test_opinion_network_models_are_created_by_name(name):
    """Test that the models are registered under their names."""
    assert create(name).components
