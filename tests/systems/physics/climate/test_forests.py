"""Test the `_forests` module of `skcomplex.systems.physics.climate`."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.systems.physics.climate import forest_dieback, social_forest_dieback

from ..conftest import _run, _series


def test_forest_dieback_settles_on_the_root_of_its_balance():
    """Test that the forest settles where `(1 - v) g(v) = gamma`, the root of Eq. 4 of Liu and Bauch."""
    root = brentq(lambda v: (1 - v) * (1 - 0.01 * (30.0 - 23 - 5 * v) ** 2) - 0.1, 0.5, 0.99)
    v = _series(_run(forest_dieback(), 200.0, 0.5), 'forest.v')[0]
    assert v[-1] == pytest.approx(root, abs=1e-6)


def test_forest_dieback_is_bistable_in_its_hysteresis_range():
    """Test that at 33.5 degrees a dense forest survives and a sparse one dies, and that at 35 degrees both die."""
    final = {
        (temperature, start): _series(_run(forest_dieback(v=start, Tv=temperature), 300.0, 0.5), 'forest.v')[0][-1]
        for temperature in (33.5, 35.0)
        for start in (0.1, 0.9)
    }
    assert final[33.5, 0.9] > 0.5
    assert final[33.5, 0.1] < 1e-3
    assert final[35.0, 0.9] < 1e-3


@pytest.mark.parametrize(('share', 'outcome'), [(0.56, 1.0), (0.53, 0.0)])
def test_social_forest_dieback_mitigation_leaves_the_unstable_share(share, outcome):
    """Test that without warming the share of mitigators leaves `x* = 1 - 2.5 / (1 + exp(3 Tc))` towards 0 or 1."""
    unstable = 1 - 2.5 / (1 + math.exp(1.5))
    assert 0.53 < unstable < 0.56
    v, x = _series(_run(social_forest_dieback(v=0.0, x=share, Tc=0.5), 200.0, 0.5), 'forest.v', 'people.x')
    assert np.abs(v).max() == 0
    assert x[-1] == pytest.approx(outcome, abs=1e-6)


def test_social_forest_dieback_keeps_the_shares_in_the_unit_interval():
    """Test that the coupled model keeps the cover and the share of mitigators between 0 and 1."""
    v, x = _series(_run(social_forest_dieback(Tv=31.0), 200.0, 0.5), 'forest.v', 'people.x')
    assert v.min() >= 0
    assert v.max() <= 1
    assert x.min() >= -1e-9
    assert x.max() <= 1 + 1e-9
