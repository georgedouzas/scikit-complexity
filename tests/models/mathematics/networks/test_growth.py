"""Test the `_growth` module of `skcomplex.models.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import FitAnalysis, NetworkAnalysis, RenormalizationAnalysis
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.models import create
from skcomplex.models.mathematics.networks import (
    barabasi_albert,
    holme_kim,
    popularity_similarity_network,
    vertex_copying,
)

from .conftest import _final_graph


def test_barabasi_albert_grows_a_scale_free_network():
    """Test that Barabási-Albert growth adds one node with two edges per step and a degree tail exponent near 3."""
    graph, trajectory = _final_graph(barabasi_albert(3003), 3000)
    assert graph.number_of_nodes() == 3003
    assert graph.number_of_edges() == 3 + 2 * 3000
    degrees = np.array([degree for _, degree in graph.degree])
    fit = FitAnalysis(minimum=8, discrete=True, bootstrap=0).check().prepare(degrees).run()
    assert abs(fit.exponent_ - 3) < 0.4
    assert np.nanmax(trajectory.values('nodes.degree')[-1]) == pytest.approx(degrees.max(), abs=2)


def test_holme_kim_keeps_clustering_that_barabasi_albert_loses():
    """Test that triad formation keeps the clustering high where preferential attachment alone lets it vanish."""
    clustered, _ = _final_graph(holme_kim(1004, triad=0.9), 1000)
    attached, _ = _final_graph(barabasi_albert(1003, edges=3, seed_nodes=3), 1000)
    assert clustered.number_of_edges() == pytest.approx(attached.number_of_edges(), rel=0.02)
    assert nx.average_clustering(clustered) > 0.3
    assert nx.average_clustering(attached) < 0.1


def test_vertex_copying_makes_hubs_that_random_attachment_lacks():
    """Test that copying edges gives a much larger largest degree than joining random nodes."""
    copied, _ = _final_graph(vertex_copying(1004, randomness=0.1), 1000)
    uniform, _ = _final_graph(vertex_copying(1004, randomness=1.0), 1000)
    largest = max(degree for _, degree in copied.degree)
    assert largest > 2 * max(degree for _, degree in uniform.degree)


@pytest.mark.parametrize('name', ['barabasi_albert', 'holme_kim', 'vertex_copying'])
def test_growth_models_are_created_by_name(name):
    """Test that the growth models are registered and start with their seed network."""
    system = create(name)
    assert system.elements[0].variable
    assert system.elements[1].on == 'edges'


def _similarity_trajectory(n_nodes: int, seed: int, **settings: float) -> Trajectory:
    """Return the trajectory of a popularity-similarity network of a size, with the frames of its start and its end."""
    system = popularity_similarity_network(n_nodes, random_state=seed, **settings)
    steps = n_nodes - 3
    return Simulation(system, dt=float(steps), random_state=seed).check().prepare().run(until=steps).trajectory_


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_popularity_similarity_network_degrees_are_heavy_tailed_and_clustered(seed: int):
    """Test that 2000 nodes of exponent 2.5 fit within 0.06 of the measured 2.70, with clustering above 0.3."""
    graph = _similarity_trajectory(2000, seed, exponent=2.5).network(-1).graph()
    fitted = float(NetworkAnalysis('degrees').check().prepare(graph).run().exponent_[0])
    assert fitted == pytest.approx(2.70, abs=0.06)
    assert nx.average_clustering(graph) > 0.3


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_popularity_similarity_network_renormalizes_geometrically_with_its_exponent(seed: int):
    """Test that the degree exponent stays within 0.2 of the original at three geometric scales."""
    trajectory = _similarity_trajectory(2000, seed, exponent=2.5)
    exponents = RenormalizationAnalysis('geometric', steps=3, beta=5.0).check().prepare(trajectory).run().exponents_
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
        popularity_similarity_network(20, **settings)
