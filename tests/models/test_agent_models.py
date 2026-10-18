"""Test the agent-based models of the social and ecology modules."""

import networkx as nx
import numpy as np

from skcomplex.models.ecology import wolf_sheep
from skcomplex.models.social import boltzmann_wealth, schelling, voter_model
from skcomplex.runtime import Simulation


def _similar_share(snapshot):
    """Return the mean share of a resident's neighbours that have its kind."""
    kinds = {
        tuple(location): kind
        for location, kind in zip(snapshot.locations.tolist(), snapshot.values['kind'], strict=True)
    }
    shares = []
    for (column, row), kind in kinds.items():
        neighbors = [
            kinds[(column + dx, row + dy)]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy) and (column + dx, row + dy) in kinds
        ]
        if neighbors:
            shares.append(np.mean([neighbor == kind for neighbor in neighbors]))
    return float(np.mean(shares))


def test_schelling_segregation_increases():
    """Test that the share of similar neighbours is higher after 100 steps than at the start."""
    trajectory = Simulation(schelling(20, 20, threshold=0.5, random_state=0), random_state=0).run(until=100).trajectory_
    assert _similar_share(trajectory.agents('residents', 100)) > _similar_share(trajectory.agents('residents', 0))


def test_boltzmann_wealth_stays_constant():
    """Test that the total wealth of all traders equals its initial value at every frame."""
    trajectory = Simulation(boltzmann_wealth(100), random_state=0).run(until=50).trajectory_
    totals = {float(trajectory.agents('traders', frame).values['wealth'].sum()) for frame in range(51)}
    assert totals == {100.0}


def test_voter_model_reaches_consensus():
    """Test that 20 voters on a connected network end with one opinion."""
    graph = nx.connected_watts_strogatz_graph(20, 4, 0.2, seed=0)
    trajectory = Simulation(voter_model(graph, random_state=0), random_state=0).run(until=500).trajectory_
    assert len(set(trajectory.agents('voters', 500).values['opinion'].tolist())) == 1


def test_wolf_sheep_populations_change():
    """Test that the wolf and sheep counts change over 100 steps and are reported every frame."""
    trajectory = Simulation(wolf_sheep(20, 20, random_state=0), random_state=0).run(until=100).trajectory_
    for group in ('sheep', 'wolves'):
        counts = trajectory[f'{group}.count']
        assert len(counts) == 101
        assert len(set(counts.tolist())) > 1
    assert set(trajectory['grass.count'].tolist()) == {400.0}
