"""Test the `_speciation` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.systems.biology.ecology import NeutralBiodiversitySystem


def test_neutral_biodiversity_species_number_matches_ewens_sum():
    """Test that the mean number of species matches S_N = sum_n theta / (theta + n) within 5%."""
    size, nu = 100, 0.02
    theta = (size - 1) * nu / (1 - nu)
    trajectory = NeutralBiodiversitySystem(size, nu=nu, random_state=0).prepare().simulate(random_state=0, until=3000)
    expected = np.sum(theta / (theta + np.arange(size)))
    assert trajectory.get_values('richness')[500:].mean() == pytest.approx(expected, rel=0.05)
