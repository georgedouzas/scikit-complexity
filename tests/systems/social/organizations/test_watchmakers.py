"""Test the `_watchmakers` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.systems.social.organizations import WatchmakersSystem

from ..conftest import _run, _values


def test_hora_finishes_watches_at_the_expected_rate_and_tempus_none():
    """Test that Hora finishes a watch every `111 ((1 - p)^-10 - 1) / p` steps on average and Tempus finishes none."""
    trajectory = _run(WatchmakersSystem(), 20000)
    finished = _values(trajectory, 'watchmakers', 'watches')
    expected = 20000 / (111 * (0.99**-10 - 1) / 0.01)
    assert finished[0] == pytest.approx(expected, rel=0.25)
    assert finished[1] == 0.0


def test_watchmakers_without_interruptions_finish_in_their_number_of_actions():
    """Test that without interruptions Tempus takes 1000 actions per watch and Hora 1110."""
    trajectory = _run(WatchmakersSystem(interruption=0.0), 5550)
    np.testing.assert_array_equal(_values(trajectory, 'watchmakers', 'watches'), [5.0, 5.0])
