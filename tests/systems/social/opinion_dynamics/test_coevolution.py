"""Test the `_coevolution` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.opinion_dynamics import CoevolvingVoterSystem, HegselmannKrauseSystem, JoinAClubSystem


def _discordant(trajectory, frame):
    """Return the share of the edges of a frame whose ends hold different opinions."""
    snapshot = trajectory.get_network(frame)
    opinions = trajectory.get_values('voters.opinion')[frame]
    return float(np.mean(opinions[snapshot.sources] != opinions[snapshot.targets]))


def test_coevolving_voter_fragments_fast_when_rewiring_dominates():
    """Test that fast rewiring removes discordant edges at once and splits opinions, while slow rewiring keeps them."""
    slow, fast = (
        simulate(CoevolvingVoterSystem(200, rewiring=rewiring).prepare().system_, until=200, dt=20, random_state=0)
        for rewiring in (0.1, 0.8)
    )
    assert _discordant(slow, 2) > 0.15
    assert _discordant(fast, 2) < 0.02
    graph = fast.get_network(-1).get_graph()
    opinions = fast.get_values('voters.opinion')[-1]
    assert nx.number_connected_components(graph) >= 2
    assert 0 < opinions.mean() < 1
    for component in nx.connected_components(graph):
        assert len({opinions[node] for node in component}) == 1


def test_coevolving_voter_keeps_the_number_of_edges():
    """Test that rewiring moves edges without creating or destroying them."""
    trajectory = simulate(CoevolvingVoterSystem(100, rewiring=0.5).prepare().system_, until=50, dt=10, random_state=1)
    sizes = {trajectory.get_network(frame).sources.size for frame in range(len(trajectory.times))}
    assert len(sizes) == 1


def test_join_a_club_separates_members_from_outsiders_at_strong_coupling():
    """Test that strong coupling cuts the acquaintances across the club boundary, and zero coupling keeps many."""
    strong = simulate(JoinAClubSystem(200, beta=1.0).prepare().system_, until=150, dt=10, random_state=0)
    free = simulate(JoinAClubSystem(200, beta=0.0).prepare().system_, until=150, dt=10, random_state=0)
    assert _discordant(strong, -1) < 0.02
    assert _discordant(free, -1) > 0.25
    members = strong.get_values('voters.opinion')[-1]
    graph = strong.get_network(-1).get_graph()
    assert nx.number_connected_components(graph) >= 2
    assert 0 < members.mean() < 1


@pytest.mark.parametrize('system', [HegselmannKrauseSystem, CoevolvingVoterSystem, JoinAClubSystem])
def test_opinion_network_systems_build_their_components(system):
    """Test that the systems build their components with their default parameters."""
    assert system().prepare().system_.elements
