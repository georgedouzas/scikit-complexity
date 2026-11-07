"""Test the `_echo_chambers` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np

from skcomplex.systems.social.opinion_dynamics import echo_chamber_radicalization

from ..conftest import _run, _values


def test_low_controversy_reaches_neutral_consensus():
    """Test that opinions converge to zero for a small controversialness."""
    opinions = _values(_run(echo_chamber_radicalization(100, controversy=0.05, homophily=2.0), 100), 'agents', 'opinion')
    assert np.max(np.abs(opinions)) < 0.01


def test_radicalization_without_homophily_is_one_sided():
    """Test that without homophily every opinion ends far out on one side."""
    opinions = _values(_run(echo_chamber_radicalization(100, homophily=0.0), 100), 'agents', 'opinion')
    assert np.all(np.sign(opinions) == np.sign(opinions[0]))
    assert np.mean(np.abs(opinions)) > 1.0


def test_homophily_polarizes_into_two_echo_chambers():
    """Test that with homophily the opinions split into two radical groups of opposite signs."""
    opinions = _values(_run(echo_chamber_radicalization(100), 100), 'agents', 'opinion')
    assert 0.3 < np.mean(opinions > 0) < 0.7
    assert np.mean(np.abs(opinions)) > 1.0
