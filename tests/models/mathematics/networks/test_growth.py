"""Test the `_growth` module of `skcomplex.models.mathematics.networks`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import fit_power_law
from skcomplex.models import create
from skcomplex.models.mathematics.networks import (
    barabasi_albert,
    holme_kim,
    vertex_copying,
)

from .conftest import _final_graph


def test_barabasi_albert_grows_a_scale_free_network():
    """Test that Barabási-Albert growth adds one node with two edges per step and a degree tail exponent near 3."""
    graph, trajectory = _final_graph(barabasi_albert(3003), 3000)
    assert graph.number_of_nodes() == 3003
    assert graph.number_of_edges() == 3 + 2 * 3000
    degrees = np.array([degree for _, degree in graph.degree])
    fit = fit_power_law(degrees, xmin=8, discrete=True, bootstrap=0)
    assert abs(fit.alpha - 3) < 0.4
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
