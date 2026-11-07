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
