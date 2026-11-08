"""Test the `_graph_spins` module of `skcomplex.systems.physics.statistical_mechanics`."""

import networkx as nx

from skcomplex.core.experiments import sweep
from skcomplex.systems.physics.statistical_mechanics import GraphIsingSystem

from .conftest import magnetization


def test_graph_ising_orders_at_low_temperature():
    """Test that the Ising system on a random 4-regular graph of 1000 nodes orders at 1.0 and not at 6.0."""
    graph = nx.random_regular_graph(4, 1000, seed=0)
    result = sweep(
        lambda temperature: GraphIsingSystem(graph, temperature=temperature, initial='up').prepare().system_,
        {'temperature': [1.0, 6.0]},
        replicates=4,
        until=400,
        dt=20,
        warmup=100,
        outputs=[magnetization],
        random_state=0,
    )
    means = {row['temperature']: row['magnetization.mean'] for row in result.summary()}
    assert means[1.0] > 0.9
    assert means[6.0] < 0.1
