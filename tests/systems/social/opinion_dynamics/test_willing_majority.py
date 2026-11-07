"""Test the `_willing_majority` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np

from skcomplex.systems.social.opinion_dynamics import willing_majority

from ..conftest import _run, _values


def test_willing_agents_reach_consensus():
    """Test that agents who all follow the majority reach consensus."""
    opinions = _values(_run(willing_majority(), 50), 'voters', 'opinion')
    assert np.all(opinions == opinions[0])


def test_zealots_of_both_opinions_prevent_consensus():
    """Test that zealots of willingness 0 holding both opinions keep the population from consensus."""
    trajectory = _run(willing_majority(zealots=0.2), 50)
    willingness = _values(trajectory, 'voters', 'willingness')
    opinions = _values(trajectory, 'voters', 'opinion')
    np.testing.assert_array_equal(
        opinions[willingness == 0],
        _values(trajectory, 'voters', 'opinion', 0)[willingness == 0],
    )
    assert 0 < opinions.mean() < 1


def test_a_small_group_of_zealots_wins_over_everyone():
    """Test that a tenth of zealots of one opinion brings every agent to it."""
    opinions = _values(_run(willing_majority(zealots=0.1, zealot_opinion=1.0), 50), 'voters', 'opinion')
    assert np.all(opinions == 1.0)
