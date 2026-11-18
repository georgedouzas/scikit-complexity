"""Test the `_rumours` module of `skcomplex.systems.social.spreading`."""

import numpy as np
import pytest

from skcomplex.systems.social.spreading import MakiThompsonSystem


def test_maki_thompson_leaves_a_fifth_ignorant():
    """Test that a rumour stops with about 20% of the population never hearing it."""
    finals = [
        MakiThompsonSystem(1000).prepare().simulate(until=40, dt=0.5, random_state=seed).get_values('ignorants.n')[-1]
        for seed in range(5)
    ]
    assert np.mean(finals) / 1000 == pytest.approx(0.203, abs=0.04)
