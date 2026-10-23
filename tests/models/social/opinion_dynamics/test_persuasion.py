"""Test the `_persuasion` module of `skcomplex.models.social.opinion_dynamics`."""

import networkx as nx
import numpy as np

from skcomplex.models.social.opinion_dynamics import (
    galam_majority,
    social_impact,
    sznajd,
)

from ..conftest import _run, _values


def test_sznajd_reaches_consensus_on_a_ring():
    """Test that the Sznajd model on a small ring ends with one opinion."""
    trajectory = _run(sznajd(nx.cycle_graph(30), random_state=0), 300)
    assert len(set(_values(trajectory, 'people', 'opinion').tolist())) == 1


def test_galam_minority_vanishes():
    """Test that a minority of 40% disappears after a few rounds of discussion in groups of three."""
    trajectory = _run(galam_majority(300, share=0.4, random_state=0), 10)
    assert _values(trajectory, 'citizens', 'opinion').sum() == 0


def test_social_impact_freezes_with_surviving_minority():
    """Test that social impact settles into a static pattern where clusters of the minority survive."""
    trajectory = _run(social_impact(random_state=2), 10)
    last, previous = _values(trajectory, 'people', 'opinion'), _values(trajectory, 'people', 'opinion', 9)
    assert np.array_equal(last, previous)
    assert 0 < (last > 0).mean() < 1


def test_social_impact_keeps_opinions_without_pressure():
    """Test that agents who all share one opinion feel no pressure to change and keep it without noise."""
    trajectory = _run(social_impact(6, 6, share=0.0, random_state=0), 5)
    assert all((_values(trajectory, 'people', 'opinion', frame) == -1).all() for frame in range(6))


def test_social_impact_flips_under_net_pressure():
    """Test that a lone dissenter surrounded by the other opinion flips to it."""
    system = social_impact(5, 5, share=0.0, random_state=0)
    opinions = np.full(25, -1.0)
    opinions[12] = 1.0
    system.elements[0].opinion = opinions
    trajectory = _run(system, 1)
    assert (_values(trajectory, 'people', 'opinion', 1) == -1).all()
