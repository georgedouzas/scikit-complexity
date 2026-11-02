"""Test the `_hypergraph_consensus` module of `skcomplex.models.social.opinion_dynamics`."""

import numpy as np

from skcomplex.models.social.opinion_dynamics import hypergraph_consensus

from ..conftest import _run


def test_hypergraph_consensus_keeps_the_mean_without_reinforcement():
    """Test that linear three-body consensus on switching triangles reaches the initial mean opinion."""
    opinions = _run(hypergraph_consensus(reinforcement=0.0), 10.0, dt=0.1).values('nodes.x')
    assert np.isclose(opinions[-1].mean(), opinions[0].mean(), atol=1e-9)
    assert opinions[-1].std() < 1e-6


def test_hypergraph_consensus_moves_the_mean_with_reinforcement():
    """Test that nonlinear three-body consensus reaches a consensus away from the initial mean opinion."""
    opinions = _run(hypergraph_consensus(reinforcement=-2.0), 10.0, dt=0.1).values('nodes.x')
    assert abs(opinions[-1].mean() - opinions[0].mean()) > 0.01
    assert opinions[-1].std() < 1e-6


def test_hypergraph_consensus_stops_while_the_triangles_are_off():
    """Test that the opinions do not move in a half period when no triangle is active."""
    system = hypergraph_consensus(period=4.0)
    system.elements[1].values['phase'][:] = np.pi
    opinions = _run(system, 2.0, dt=0.1).values('nodes.x')
    assert np.allclose(opinions[-1], opinions[0])
