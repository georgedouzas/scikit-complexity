"""Test the `_growth` module of `skcomplex.systems.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import fit_law, network_measures, renormalization
from skcomplex.core.engine.runtime import Trajectory, simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.mathematics.networks import (
    BarabasiAlbertSystem,
    GraphletArrivalSystem,
    HolmeKimSystem,
    PopularitySimilarityNetworkSystem,
    VertexCopyingSystem,
)

from .conftest import _final_graph


def test_barabasi_albert_grows_a_scale_free_network():
    """Test that Barabási-Albert growth adds one node with two edges per step and a degree tail exponent near 3."""
    graph, trajectory = _final_graph(BarabasiAlbertSystem(3003), 3000)
    assert graph.number_of_nodes() == 3003
    assert graph.number_of_edges() == 3 + 2 * 3000
    degrees = np.array([degree for _, degree in graph.degree])
    fit = fit_law(degrees, minimum=8, discrete=True, bootstrap=0)
    assert abs(fit.exponent - 3) < 0.4
    assert np.nanmax(trajectory.values('nodes.degree')[-1]) == pytest.approx(degrees.max(), abs=2)


def test_holme_kim_keeps_clustering_that_barabasi_albert_loses():
    """Test that triad formation keeps the clustering high where preferential attachment alone lets it vanish."""
    clustered, _ = _final_graph(HolmeKimSystem(1004, triad=0.9), 1000)
    attached, _ = _final_graph(BarabasiAlbertSystem(1003, edges=3, seed_nodes=3), 1000)
    assert clustered.number_of_edges() == pytest.approx(attached.number_of_edges(), rel=0.02)
    assert nx.average_clustering(clustered) > 0.3
    assert nx.average_clustering(attached) < 0.1


def test_vertex_copying_makes_hubs_that_random_attachment_lacks():
    """Test that copying edges gives a much larger largest degree than joining random nodes."""
    copied, _ = _final_graph(VertexCopyingSystem(1004, randomness=0.1), 1000)
    uniform, _ = _final_graph(VertexCopyingSystem(1004, randomness=1.0), 1000)
    largest = max(degree for _, degree in copied.degree)
    assert largest > 2 * max(degree for _, degree in uniform.degree)


@pytest.mark.parametrize('system_class', [BarabasiAlbertSystem, HolmeKimSystem, VertexCopyingSystem])
def test_growth_systems_start_with_their_seed_network(system_class):
    """Test that the growth systems start with their seed network."""
    system = system_class().prepare().system_
    assert system.elements[0].variable
    assert system.elements[1].on == 'edges'


def _similarity_trajectory(n_nodes: int, seed: int, **settings: float) -> Trajectory:
    """Return the trajectory of a popularity-similarity network of a size, with the frames of its start and its end."""
    system = PopularitySimilarityNetworkSystem(n_nodes, random_state=seed, **settings)
    steps = n_nodes - 3
    return simulate(system, until=steps, dt=float(steps), random_state=seed)


def test_popularity_similarity_network_degrees_are_heavy_tailed_and_clustered():
    """Test that 2000 nodes of exponent 2.5 fit a mean over seeds within 0.07 of 2.69, and are clustered."""
    graphs = [_similarity_trajectory(2000, seed, exponent=2.5).network(-1).graph() for seed in range(4)]
    fitted = [float(network_measures(graph, kind='degrees').exponent[0]) for graph in graphs]
    assert np.mean(fitted) == pytest.approx(2.69, abs=0.07)
    assert min(nx.average_clustering(graph) for graph in graphs) > 0.3


def test_popularity_similarity_network_renormalizes_geometrically_with_its_exponent():
    """Test that the mean degree exponent over seeds stays within 0.2 of the original at three geometric scales."""
    exponents = np.mean(
        [
            renormalization(
                _similarity_trajectory(2000, seed, exponent=2.5),
                kind='geometric',
                steps=3,
                beta=5.0,
            ).exponents
            for seed in range(4)
        ],
        axis=0,
    )
    assert np.abs(exponents[1:] - exponents[0]).max() <= 0.2


def test_popularity_similarity_network_new_nodes_join_at_close_angles():
    """Test that a low temperature joins a new node mostly to the angles near it, which a uniform draw would not."""
    trajectory = _similarity_trajectory(400, 0, temperature=0.1)
    network = trajectory.network(-1)
    angles = np.asarray(trajectory.values('nodes.angle')[-1])
    gaps = np.abs(angles[network.sources] - angles[network.targets])
    gaps = np.minimum(gaps, 2 * np.pi - gaps)
    assert np.median(gaps) < np.pi / 4


def test_popularity_similarity_network_clustering_falls_with_the_temperature():
    """Test that a higher temperature lowers the clustering."""
    low = nx.average_clustering(_similarity_trajectory(500, 0, temperature=0.1).network(-1).graph())
    high = nx.average_clustering(_similarity_trajectory(500, 0, temperature=0.9).network(-1).graph())
    assert low > high + 0.1


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'exponent': 2.0}, 'degree exponent'),
        ({'temperature': 0.0}, 'temperature'),
        ({'temperature': 1.0}, 'temperature'),
    ],
)
def test_popularity_similarity_network_rejects_invalid_settings(settings: dict[str, float], message: str):
    """Test that an exponent not above 2 and a temperature outside the unit interval raise an error."""
    with pytest.raises(ValidationError, match=message):
        PopularitySimilarityNetworkSystem(20, **settings).prepare()


def test_bianconi_barabasi_fitness_lets_fit_nodes_win_edges():
    """Test that with fitness the degree of the older nodes rises with their fitness, and every step adds two edges."""
    graph, trajectory = _final_graph(BarabasiAlbertSystem(1003, fitness=True), 1000)
    assert graph.number_of_edges() == 3 + 2 * 1000
    fitness, age, degree = (trajectory.values(f'nodes.{state}')[-1] for state in ('fitness', 'age', 'degree'))
    old = age > 500
    assert np.corrcoef(np.argsort(np.argsort(fitness[old])), np.argsort(np.argsort(degree[old])))[0, 1] > 0.4
    assert fitness.min() >= 0
    assert fitness.max() <= 1


@pytest.mark.parametrize('alpha', [0.0, 0.3, 1.0])
def test_graphlet_arrival_grows_a_tree_with_the_degree_shares_of_filkov(alpha):
    """Test that graphlets grow a tree with the degree shares `2 (2 - alpha) / 5` and `alpha (7 - alpha) / 15`."""
    graph, _ = _final_graph(GraphletArrivalSystem(2002, alpha=alpha), 1000)
    assert nx.is_tree(graph)
    assert graph.number_of_nodes() == 2002
    degrees = np.array([degree for _, degree in graph.degree])
    assert np.mean(degrees == 1) == pytest.approx(2 * (2 - alpha) / 5, abs=0.03)
    assert np.mean(degrees == 2) == pytest.approx(alpha * (7 - alpha) / 15, abs=0.03)
