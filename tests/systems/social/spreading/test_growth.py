"""Test the `_growth` module of `skcomplex.systems.social.spreading`."""

import numpy as np
import pytest

from skcomplex.systems.social.spreading import SimonYuleSystem

from ..conftest import _run, _values


def test_simon_yule_grows_new_words_at_the_innovation_rate():
    """Test that the text has its expected length and about `innovation` times as many distinct words."""
    counts = _values(_run(SimonYuleSystem(innovation=0.1), 100), 'words', 'count')
    assert counts.sum() == 10_001
    assert len(counts) == pytest.approx(1000, rel=0.1)
    assert counts.max() > 100 * np.median(counts)


def test_simon_yule_rank_slope_is_one_minus_innovation():
    """Test that the counts of the top words fall with the rank slope `1 - innovation`."""
    counts = np.sort(_values(_run(SimonYuleSystem(innovation=0.5, words_per_step=1000), 20), 'words', 'count'))[::-1]
    ranks = np.arange(1, 101)
    slope = -np.polyfit(np.log(ranks), np.log(counts[:100]), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.1)
