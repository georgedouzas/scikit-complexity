"""Test the `_typewriting` module of `skcomplex.systems.social.language`."""

import numpy as np
import pytest

from skcomplex.systems.social.language import random_typewriting

from ..conftest import _run, _values


def test_word_frequency_falls_exponentially_with_length():
    """Test that a word of `L + 1` letters is `letters / (1 - q)` times rarer than a word of `L` letters."""
    trajectory = _run(random_typewriting(), 100)
    counts, lengths = _values(trajectory, 'words', 'count'), _values(trajectory, 'words', 'length')
    assert np.sum(lengths == 1) == 5
    assert np.sum(lengths == 2) == 25
    ratio = counts[lengths == 1].mean() / counts[lengths == 2].mean()
    assert ratio == pytest.approx(5 / 0.8, rel=0.1)


def test_rank_frequency_is_a_power_law():
    """Test that the frequency of the word of rank `r` falls as `r^(-alpha)` with `alpha = 1 - log(1 - q) / log N`."""
    trajectory = _run(random_typewriting(), 100)
    counts, lengths = _values(trajectory, 'words', 'count'), _values(trajectory, 'words', 'length')
    ranks = np.cumsum(5.0 ** np.arange(1, 5))
    frequencies = [counts[lengths == length].mean() for length in range(1, 5)]
    slope = np.polyfit(np.log(ranks[1:]), np.log(frequencies[1:]), 1)[0]
    assert -slope == pytest.approx(1 - np.log(0.8) / np.log(5), abs=0.1)
