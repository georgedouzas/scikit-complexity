"""Test the `_distribution_networks` module of `skcomplex.systems.biology.physiology`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.physiology import WestBrownEnquistSystem


def _totals(depth: int) -> tuple[float, float]:
    """Return the blood volume and the number of capillaries of a tree of a depth."""
    trajectory = simulate(WestBrownEnquistSystem(depth), until=1.0)
    return float(trajectory.get_values('blood_volume')[-1]), float(trajectory.get_values('capillaries')[-1])


@pytest.mark.parametrize('depth', [3, 5])
def test_volume_and_capillaries_follow_the_branching_rules(depth):
    """Test that the volume and the capillaries of the tree are the sums of West, Brown, and Enquist."""
    volume, capillaries = _totals(depth)
    assert capillaries == 2**depth
    expected = math.pi * 2**depth * (2 ** (depth / 3) - 1) / (2 ** (1 / 3) - 1)
    assert volume == pytest.approx(expected, rel=1e-9)


def test_metabolic_exponent_approaches_three_quarters_from_below():
    """Test that the exponent between consecutive trees rises towards Kleiber's 3/4."""
    totals = np.array([_totals(depth) for depth in (3, 4, 5, 6)])
    exponents = np.diff(np.log(totals[:, 1])) / np.diff(np.log(totals[:, 0]))
    assert np.all(np.diff(exponents) > 0)
    assert 0.65 < exponents[-1] < 0.75
